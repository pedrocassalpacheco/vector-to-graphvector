from abc import ABC, abstractmethod
from typing import Iterable
from langchain_core.documents import Document

class DocumentProcessor(ABC):
    
    def process(self, documents: Iterable[Document]) -> Iterable[Document]:
        logger.info("Creating content graph from existing langchain documents ...")
        self.infer_hierarchy = infer_hierarchy
        self.graph.clear() if reset_graph else None

        self.infered_parent = self.root = Document(
            id="root",
            page_content=self.name,
            metadata={"file_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        )
        self.graph.append(self.root)
        
        # Vial hack - need access to documents while infering hiearchy
        self.original_documents = documents.copy()
        for doc in documents:
            try:
                # Extract the element type from the class name
                element_type = doc.metadata["type"]
                doc = self.handle_hierarchy(element_type, doc)
                self.graph.append(doc)

            except Exception as e:
                logger.error(
                    f"An error occurred while processing element {doc.id}: {e}"
                )
                logger.error(e, exc_info=True)
                break

        return None
        pass
    
    def _add_hierarchy(self, document: Document) -> Document:
        pass
    
class unstructuredElementStrategyProcessor(DocumentProcessor):
    def process(self, documents: Iterable[Document]) -> Iterable[Document]:
        print("Processing text documents")
        return documents
    

class TitleStrategyProcessor(DocumentProcessor):
    def process(self, documents: Iterable[Document]) -> Iterable[Document]:
        print("Processing image documents")
        return documents

class PDFStrategyProcessor(DocumentProcessor):
    def process(self, documents: Iterable[Document]) -> Iterable[Document]:
        print("Processing PDF documents")
        return documents

# Mapping document types to processor classes
processor_map = {
    "element": ElementStrategyProcessor,
    "title": TitleStrategyProcessor,
    "pdf": PDFStrategyProcessor
}

def get_processor(document_type: str) -> DocumentProcessor:
    processor_class = processor_map.get(document_type)
    if processor_class is None:
        raise ValueError("Unknown document type")
    return processor_class()

# Example usage
document_type = "element"
processor = get_processor(document_type)
documents = [Document(), Document()]  # Example documents
processed_documents = processor.process(documents)