import os
import json
import sys
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import ragas
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.run_config import RunConfig
from app.config import get_settings


# configuration
load_dotenv(PROJECT_ROOT / ".env")
settings = get_settings()

DOCUMENT_FOLDER = PROJECT_ROOT / "eval" / "test_docs"
OUTPUT_FOLDER = PROJECT_ROOT / "eval" / "output"
CACHE_FOLDER = PROJECT_ROOT / "eval" / "cache"

TESTSET_SIZE = 3

# Model used only for dataset generation (overrides settings on purpose).
OLLAMA_LLM_MODEL = "qwen2.5:7b-instruct"
OLLAMA_MODEL = settings.ollama_embed_model
OLLAMA_URL = settings.ollama_base_url

# Cache files
KG_CACHE_PATH = CACHE_FOLDER / "knowledge_graph.json"
KG_MANIFEST_PATH = CACHE_FOLDER / "knowledge_graph_manifest.json"

TESTSET_CSV_PATH = OUTPUT_FOLDER / "evaluation_dataset.csv"
TESTSET_JSON_PATH = OUTPUT_FOLDER / "evaluation_dataset.json"
TESTSET_MANIFEST_PATH = OUTPUT_FOLDER / "evaluation_dataset_manifest.json"

CACHE_VERSION = 1


# document loader
def load_document(file_path: Path) -> list[Document]:

    extension = file_path.suffix.lower()
    documents = []

    print(f"Loading: {file_path.name}")

    if extension == ".pdf":

        loader = PyMuPDFLoader(str(file_path))
        documents = loader.load()

    elif extension == ".docx":

        from docx import Document as WordDocument

        doc = WordDocument(str(file_path))

        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )

        # Include table content
        for table in doc.tables:
            for row in table.rows:
                text += "\n" + " | ".join(
                    cell.text for cell in row.cells
                )

        documents = [
            Document(
                page_content=text,
                metadata={"source": str(file_path)}
            )
        ]

    elif extension == ".pptx":

        from pptx import Presentation

        presentation = Presentation(str(file_path))

        for slide_number, slide in enumerate(
            presentation.slides, start=1
        ):

            slide_text = []

            for shape in slide.shapes:

                if shape.has_text_frame:
                    slide_text.append(shape.text)

                if shape.has_table:
                    for row in shape.table.rows:
                        slide_text.append(
                            " | ".join(
                                cell.text for cell in row.cells
                            )
                        )

            content = "\n".join(slide_text)

            if content.strip():
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "page": slide_number
                        }
                    )
                )

    elif extension in [".txt", ".md"]:

        text = file_path.read_text(encoding="utf-8")

        documents = [
            Document(
                page_content=text,
                metadata={"source": str(file_path)}
            )
        ]

    else:
        print(f"Unsupported format: {extension}")
        return []

    return [
        doc for doc in documents
        if doc.page_content.strip()
    ]


# Generate a fingerprint to detect changes in documents or models
def generate_fingerprint(all_documents: list[Document]) -> str:

    document_data = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in all_documents
    ]

    fingerprint_data = {
        "documents": document_data,
        "llm_model": OLLAMA_LLM_MODEL,
        "ollama_embedding_model": OLLAMA_MODEL,
        "ragas_version": ragas.__version__,
        "cache_version": CACHE_VERSION,
    }

    serialized = json.dumps(
        fingerprint_data,
        ensure_ascii=False,
        sort_keys=True,
        default=str
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# Save JSON metadata
def save_manifest(path: Path, data: dict):

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    temporary_path.replace(path)


# Fail-safe helpers
def make_filters_safe(transform):
    """If a CustomNodeFilter fails to score a node, keep the node instead of crashing."""
    if isinstance(transform, CustomNodeFilter):
        original_filter = transform.custom_filter

        async def safe_filter(node, kg):
            try:
                return await original_filter(node, kg)
            except Exception as e:
                print(f"Filter scoring failed, keeping node: {type(e).__name__}")
                return False  # True means "remove node", so False keeps it

        transform.custom_filter = safe_filter

    for child in getattr(transform, "transformations", None) or []:
        make_filters_safe(child)


def make_headlines_safe(transform):
    """If headline extraction fails for a node, store an empty list and move on."""
    original_extract = transform.extract

    async def safe_extract(node):
        try:
            return await original_extract(node)
        except Exception as e:
            print(f"Headline extraction failed, skipping node: {type(e).__name__}")
            return "headlines", []

    transform.extract = safe_extract


# Setup the local Ollama LLM and Nomic embeddings.
print(f"\nInitializing Ollama LLM: {OLLAMA_LLM_MODEL}")

qwen = ChatOllama(
    model=OLLAMA_LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.2,
    repeat_penalty=1.1,
    num_ctx=8192,
    num_predict=2048,
)

generator_llm = LangchainLLMWrapper(
    qwen,
    run_config=RunConfig(timeout=600, max_retries=5, max_workers=2),
)

print("\nInitializing Nomic embeddings")
nomic = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
)

generator_embeddings = LangchainEmbeddingsWrapper(nomic)


# Load documents
print("\nReading documents")

all_documents = []

document_files = sorted(
    file_path
    for file_path in DOCUMENT_FOLDER.rglob("*")
    if file_path.is_file()
)[:5]

for file_path in document_files:

    loaded = load_document(file_path)
    all_documents.extend(loaded)


if not all_documents:
    raise ValueError("No readable documents found.")


print(f"\nTotal document sections: {len(all_documents)}")


# Generate fingerprint
document_fingerprint = generate_fingerprint(all_documents)


# setup the testset generator
print("\nInitializing RAGAS Testset Generator")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


# Check whether the testset already exists
testset_cache_valid = False

if (
    TESTSET_CSV_PATH.exists()
    and TESTSET_JSON_PATH.exists()
    and TESTSET_MANIFEST_PATH.exists()
):

    with open(TESTSET_MANIFEST_PATH, "r", encoding="utf-8") as file:
        testset_manifest = json.load(file)

    if (
        testset_manifest.get("fingerprint") == document_fingerprint
        and testset_manifest.get("testset_size") == TESTSET_SIZE
    ):
        testset_cache_valid = True


if testset_cache_valid:

    print("\nExisting testset found. Loading cached dataset.")

    df = pd.read_csv(TESTSET_CSV_PATH)

    print(f"Loaded cached test cases: {len(df)}")

else:

    # Setup the knowledge graph
    print("\nPreparing RAGAS Knowledge Graph")

    kg_manifest = None

    if KG_MANIFEST_PATH.exists():

        with open(KG_MANIFEST_PATH, "r", encoding="utf-8") as file:
            kg_manifest = json.load(file)

    kg_cache_valid = (
        KG_CACHE_PATH.exists()
        and kg_manifest is not None
        and kg_manifest.get("fingerprint") == document_fingerprint
    )

    if kg_cache_valid:

        print("\nLoading cached Knowledge Graph")

        kg = KnowledgeGraph.load(str(KG_CACHE_PATH))

        completed_transforms = kg_manifest.get(
            "completed_transforms", 0
        )

        print(f"Loaded nodes: {len(kg.nodes)}")
        print(f"Loaded relationships: {len(kg.relationships)}")
        print(f"Completed transformations: {completed_transforms}")

    else:

        print("\nCreating new Knowledge Graph")

        nodes = [
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
            for doc in all_documents
        ]

        kg = KnowledgeGraph(nodes=nodes)
        completed_transforms = 0

        save_manifest(
            KG_MANIFEST_PATH,
            {
                "fingerprint": document_fingerprint,
                "completed_transforms": 0,
            }
        )

    # Prepare transformations
    transforms = default_transforms(
        all_documents,
        generator_llm,
        generator_embeddings,
    )

    for transform in transforms:
        transform_name = transform.__class__.__name__

        # Resume HeadlinesExtractor at node level: nodes that already have
        # "headlines" were processed in an earlier attempt.
        if transform_name == "HeadlinesExtractor":
            transform.filter_nodes = lambda node: (
                node.type == NodeType.DOCUMENT
                and node.get_property("headlines") is None
            )
            make_headlines_safe(transform)

        # HeadlineSplitter only works on document nodes whose headlines
        # have already been extracted.
        elif transform_name == "HeadlineSplitter":
            transform.filter_nodes = lambda node: (
                node.type == NodeType.DOCUMENT
                and node.get_property("headlines") is not None
            )

        make_filters_safe(transform)

    # Apply transformations and checkpoint after each one
    if completed_transforms < len(transforms):

        print("\nGenerating document transformations")

        for index in range(completed_transforms, len(transforms)):

            transform = transforms[index]
            transform_name = transform.__class__.__name__

            print(
                f"\nApplying transformation "
                f"{index + 1}/{len(transforms)}: {transform_name}"
            )

            try:
                apply_transforms(kg, [transform])

                if transform_name == "HeadlinesExtractor":
                    missing = [
                        n for n in kg.nodes
                        if n.type == NodeType.DOCUMENT
                        and n.get_property("headlines") is None
                    ]
                    if missing:
                        raise RuntimeError(
                            f"{len(missing)} nodes still missing headlines; "
                            "rerun to resume."
                        )

            except Exception:
                # Save whatever node properties completed so the next run
                # can resume instead of repeating them.
                print(
                    "\nTransformation failed. Saving partial Knowledge "
                    "Graph checkpoint before stopping..."
                )
                kg.save(str(KG_CACHE_PATH))
                save_manifest(
                    KG_MANIFEST_PATH,
                    {
                        "fingerprint": document_fingerprint,
                        "completed_transforms": index,  # keep this one pending
                    }
                )
                print("Partial Knowledge Graph checkpoint saved.")
                raise

            # Save Knowledge Graph immediately after successful transformation
            kg.save(str(KG_CACHE_PATH))

            save_manifest(
                KG_MANIFEST_PATH,
                {
                    "fingerprint": document_fingerprint,
                    "completed_transforms": index + 1,
                }
            )

            print("Knowledge Graph checkpoint saved.")

    else:

        print("\nAll transformations already completed.")

    print("\nKnowledge Graph ready.")
    print(f"Nodes: {len(kg.nodes)}")
    print(f"Relationships: {len(kg.relationships)}")

    # Initialize generator using the cached Knowledge Graph
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
    )

    # generate the evaluation dataset
    print("\nGenerating evaluation dataset")

    testset = generator.generate(
        testset_size=TESTSET_SIZE
    )

    df = testset.to_pandas()

    # export result
    df.to_csv(TESTSET_CSV_PATH, index=False)

    df.to_json(
        TESTSET_JSON_PATH,
        orient="records",
        indent=4,
        force_ascii=False
    )

    # Save testset manifest
    save_manifest(
        TESTSET_MANIFEST_PATH,
        {
            "fingerprint": document_fingerprint,
            "testset_size": TESTSET_SIZE,
            "llm_model": OLLAMA_LLM_MODEL,
            "ragas_version": ragas.__version__,
        }
    )

    print("\nTestset cache saved.")


print("\n===================================")
print("       GENERATION COMPLETED")
print("===================================")

print(f"Generated test cases: {len(df)}")
print(f"CSV:  {TESTSET_CSV_PATH}")
print(f"JSON: {TESTSET_JSON_PATH}")

print("\nDataset preview:")
print(df.head())