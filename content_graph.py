"""Helper class to generate Document based graph for graph vector store."""

import json
import logging
import json
import zlib
import base64
from datetime import datetime
from pathlib import Path
import uuid

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union
from enum import Enum

from pyvis.network import Network

from unstructured.documents.elements import Element

from langchain.schema import Document
from langchain_community.graph_vectorstores.networkx import documents_to_networkx
from langchain_community.graph_vectorstores.links import Link, add_links, get_links
from langchain_astradb.utils.vector_store_codecs import (
    _AstraDBVectorStoreDocumentCodec,
    _DefaultVectorizeVSDocumentCodec,
    _DefaultVSDocumentCodec,
)
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyType(Enum):
    BASIC = "basic"
    BY_ELEMENT = "by_element"
    BY_TITLE = "by_title"
    BY_PAGE = "by_page"
    SEMANTIC = "by_similarity"

#region Helper Functions
def _upgrade_documents(
    vector_store: "AstraDBVectorStore",
    batch_size: int = 10,
    documents: List["Document"] = []
) -> List["Document"]:
    """
    Upgrade documents in the vector store by adding an "upgraded" metadata field.
    Args:
        vector_store (AstraDBVectorStore): The vector store instance to update.
        batch_size (int, optional): The number of documents to process in each batch. Defaults to 10.
        documents (List[Document], optional): The list of documents to upgrade. Defaults to an empty list.
    Returns:
        List[Document]: The list of original documents that were upgraded.
    """

    filter = {"upgraded": {"$exists": False}}
   
    if len(chunks) == 0:
        return 0

    id_to_md_map: dict[str, dict] = {}
    original_documents = []
    for chunk in chunks:
        original_documents.append(chunk)
        chunk.metadata["upgraded"] = True
        id_to_md_map[chunk.id] = chunk.metadata
    
        vector_store.update_metadata(id_to_md_map)
        
    return original_documents

def _explore_composite_elements(original_documents: List[Document]) -> List[Document]:
    """Explores and processes composite elements within a list of original documents.
    This function takes a list of original documents, restores their original elements,
    and processes each element to create new documents with updated metadata. It prints
    detailed information about the processing steps and the elements being handled.
    
    Args:
        original_documents (List[Document]): A list of Document objects to be processed.
    Returns:
        List[Document]: A list of new Document objects with updated metadata.
    """

    logger.info(f"Original document has {len(original_documents)} parts")
    converted_documents = []
    for original_document in original_documents:
        logger.info(f"\n\nProcessing document {original_document.id} of type {original_document.metadata['type']}")

        original_elements = restore_original_elements(original_document)    
        logger.info(f"\tFound {len(original_elements)} elements")
        for original_element in original_elements:
            print("--------------------")
            print(f"\tProcessing original element id {original_element['element_id']}")
            new_metadata = {
                "parent_id": original_element["metadata"].get("parent_id",None),
            }
            
            logger.info(f"\t\tType {original_element['type']}")
            logger.info(f"\t\tParent ID {original_element['metadata'].get('parent_id',None)}")
            
            parent_ids.append(new_metadata["parent_id"])
            updated_metadata = {**original_document.metadata, **new_metadata}
            
            new_document = Document(
                id=original_document.metadata["element_id"],
                page_content=original_document.page_content,
                metadata=updated_metadata
            )
            converted_documents.append(new_document)
        return converted_documents

def _document_to_dict(doc: Document) -> dict:
    """Converts a document object to a dictionary representation. Mostly used for debugging purposes.

    Args:
        doc: The document object to convert. It is expected to have `metadata` and `page_content` attributes.

    Returns:
        dict: A dictionary containing the following keys:
            - "element_id" (str or None): The element ID from the document's metadata.
            - "page_content" (str): The content of the document's page.
            - "type" (str): The type of the document from its metadata.
            - "parent_id" (str or None): The parent ID from the document's metadata.
            - "parent_found" (bool): A boolean indicating if the parent document was found by its element ID.
    """
    return {
        "element_id": doc.metadata.get("element_id", None),
        "page_content": doc.page_content,
        "type": doc.metadata["type"],
        "parent_id": doc.metadata["metadata"].get("parent_id", None),
        "parent_found": find_document_by_element_id(doc.metadata["metadata"].get("parent_id", None)),
    }

def _encode_astradb_documents(hits: List[dict]) -> List[Document]:
    """Encodes a list of AstraDB documents into a list of Document objects.

    Args:
        hits (List[dict]): A list of dictionaries representing AstraDB documents.

    Returns:
        List[Document]: A list of Document objects decoded from the input dictionaries.
    """
    document_codec = _DefaultVSDocumentCodec(content_field="content",ignore_invalid_documents=True)
    return [document_codec.decode(hit) for hit in hits]

#endregion

@dataclass
class ContentGraph:
    def __init__(
        self, name: str = "Content Graph", metadata: Optional[Dict[str, Union[str, int]]] = None
    ) -> None:
        """
        Initializes a ContentGraph instance.

        :param title: Title of the content graph.
        :param metadata: Optional metadata dictionary with string keys and values that can be strings or integers.
        """
        self.name: str = name
        self.graph: List[Document] = []  # Store nodes as LangChain Documents
        self.root: Document = None
        self.infered_parent: Document = None
        self.infer_hierarchy: bool = True
        self._strategy_handlers = {
            StrategyType.BASIC: self._basic_strategy_handler,
            StrategyType.BY_ELEMENT: self._element_strategy_handler,
            StrategyType.BY_TITLE: self._title_strategy_handler,
        }
        self.documents_bucket = []
        self.uuid_dict = {}
        
    def fromPDFDocument(
        self,
        documents: List[Document],
        output_image_path: Path,
        reset_graph: bool = False,
        infer_hierarchy: bool = True,
        strategy: str = "element",
    ) -> None:
        """
        Synchronously processes a PDF document.

        :param pdf_path: The path to the PDF file.
        :param pdf_path: The path to were images are stored.
        """
        logger.info(f"Synchronously processing PDF document from '{file_path}'...")
        self.infer_hierarchy = infer_hierarchy
        self.graph.clear() if reset_graph else None

        self.infered_parent = self.root = Document(
            id="root",
            page_content=str(file_path),
            metadata={"file_date": datetime.fromtimestamp(file_path.stat().st_ctime)},
        )
        self.graph.append(self.root)

        if not file_path.exists() or not file_path.is_file():
            logger.error(f"File at {file_path} does not exist or is not a valid file.")
            return None

        elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            max_characters=2000,
            new_after_n_chars=1700,
            extract_image_block_output_dir="images/",
        )

        for element in elements:
            try:
                # Extract the element type from the class name
                element_type = type(element).__name__
                doc = self.element_to_document(element)
                self.graph.append(self.handle_hierarchy(element_type, doc))

            except Exception as e:
                logger.error(
                    f"An error occurred while processing element {element.id}"
                )
                logger.error(e, exc_info=True)
        return None

    def fromLangChainDocuments(
        self,  
        documents: List[Document],
        name: str = "content_graph",
        reset_graph: bool = False,
        infer_hierarchy: bool = True,
        deserialize_links: bool = False,
        strategy: StrategyType = StrategyType.BY_ELEMENT 
    ) -> None:
        """
        Synchronously processes langchain Documents into a content graph.

        :param pdf_path: The path to the PDF file.
        :param pdf_path: The path to were images are stored.
        """
        logger.info("Creating content graph from existing langchain documents ...")
        self.infer_hierarchy = infer_hierarchy
        self.graph.clear() if reset_graph else None
        self.documents_bucket = documents
        
        # Starts from a root element
        self.infered_parent = self.root = Document(
            id="root",
            page_content=self.name,
            metadata={"file_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        )
        self.graph.append(self.root)
        strategy_enum = StrategyType[strategy.upper()]
        handler = self._strategy_handlers.get(strategy_enum)
        
        for doc in documents:
            try:
                doc = handler(doc)
                self.graph.append(doc)

            except Exception as e:
                logger.error(
                    f"An error occurred while processing element {doc.id}: {e}"
                )
                logger.error(e, exc_info=True)

        return None

    #region Private Methods
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def _element_to_document(self, element: Element) -> Document:
        
        return Document(
            id=element["element_id"],
            page_content=element["text"],
            metadata={
                "type": element["type"],
                "links": [],
                "metadata": element["metadata"],
            },
        )

    def _add_link(self, doc: Document, link:Link) -> Document:    
        """Adds a link to a document's metadata ensuring that the link is unique. This is needed because the based function allows for multiple identical links to be added
        
        Args:
            doc (Document): The document to which the link is added.
            link (Link): The link to add.

        Returns:
            Document: The updated document with the link added.

        """
        existing_links = get_links(doc)
        
        for existing_link in existing_links:
            if  existing_link.direction == link.direction and \
                existing_link.kind == link.kind and \
                existing_link.tag == link.tag:
                
                # Nothing to do because the link already exists
                return doc
        
        add_links(doc, link)
        return doc
    
    def _restore_original_elements(self, doc: Document) -> dict:
        """
        Restores the original elements from the given document's metadata.

        This function decodes a base64-encoded and zlib-compressed JSON string
        stored in the document's metadata under the key "orig_elements". It then
        parses the JSON string into a dictionary and returns it.

        Args:
            doc (Document): The document containing the metadata with the original elements.

        Returns:
            dict: A dictionary representing the original elements.
        """
        decoded_b64_bytes = base64.b64decode(doc.metadata["metadata"]["orig_elements"])
        elements_json_bytes = zlib.decompress(decoded_b64_bytes)
        elements_json_str = elements_json_bytes.decode("utf-8")
        element_dicts = json.loads(elements_json_str)
        return element_dicts

    # region Strategy Handlers
    def _element_strategy_handler(self, doc: Document) -> Document:
        """Figure out the hierarchy of the document based on the element type. Please see langchain_community/graph_vectorstores/link.py for details."""
        
        element_id = doc.metadata["element_id"] or None
        element_type = doc.metadata["type"] or None
        parent_id = doc.metadata["metadata"].get("parent_id", None)

        if not element_type:
            raise Exception("Element type not found in document metadata")

        if element_type == "Title":
            tag = "0"
            # From root to title
            outgoing_link = Link.outgoing(kind=element_type, tag=tag)
            self._add_link(self.root, outgoing_link)
           
            # From title to root
            self._add_link(doc, Link.incoming(kind="root", tag=tag))
        
            # Records last title so it can be a parent to the next element
            self.infered_parent = doc

        elif self.infer_hierarchy is False and parent_id is not None:
            # From parent to whatever this is
            parent_doc = self.find_document_by_element_id(parent_id)
            if parent_doc is None:
                raise Exception(f'Parent document not found for {doc.id}')

            tag = parent_id
            self._add_link(parent_doc, Link.outgoing(kind=element_type, tag=tag))

            # From whatever this is to parent
            self._add_link(doc, Link.incoming(kind=element_type, tag=tag))

        else:
            # If there is no parent, link to the last saved possible parent
            tag = self.infered_parent.metadata.get("element_id", None)
            self._add_link(self.infered_parent, Link.outgoing(kind=element_type, tag=tag))

            # From whatever this is to parent
            self._add_link(doc, Link.incoming(kind=element_type, tag=tag))
            
        return doc

    def _title_strategy_handler(self, doc: Document) -> Document:
        """This particular strategy, the top level elements are of the type composite. The composite element is what gets 
        embedded, but it is very corse grained. So the strategy is to break apart an encoded string which contains all the 
        original elements, using the compositive elements are the direct descended of the root, and the original elements are descendeds 
        of the composite elemenets
        """
        
        element_id = doc.metadata["element_id"] or None
        element_type = doc.metadata["type"] or None
        parent_id = doc.metadata["metadata"].get("parent_id", None)

        if not element_type and element_type != "CompositeElement":
            raise Exception("Invalid element type for title strategy. Excted CompositeElement, but got {element_type}")

        # Connects element to root
        tag = "0"
        
        # From root to title
        self._add_link(self.root, Link.outgoing(kind=element_type, tag=tag))

        # From title to root
        self._add_link(doc, Link.incoming(kind="root", tag=tag))

        # Now iterate over the original elements and connect them to the graph
        original_elements = self._restore_original_elements(doc)
        for original_element in original_elements:
            
            # Adds all original elements to the composite element. There may be a more fine grained way to do this however, the individual elements are too fined grained to justify embedding them in the graph
            inner_document = self._element_to_document(original_element)
            
            # From composite to inner document 
            self._add_link(doc, Link.outgoing(kind=element_type, tag=element_id))
            
            # And vice versa
            self._add_link(inner_document, Link.incoming(kind=element_type, tag=element_id))

            # Adds the inner document to the graph. TODO: This is a bit redundat. IF anything else, maybe original elements should removed from the composite element to save space
            self.graph.append(inner_document)

        return doc

    def _basic_strategy_handler(self, doc: Document) -> Document:
        pass

    def _page_strategy_handler(self, doc: Document) -> Document:
        pass

    def _semantic_strategy_handler(self, doc: Document) -> Document:
        pass
    #endregion
    
    # region Graph Navigation
    def find_document_by_element_id(self, element_id: str) -> Document:
        return next((d for d in self.graph if d.metadata.get("element_id", None) == element_id), None)
    
    def find_children_by_parent_id(self, parent_id: str) -> List[Document]:
        return [d for d in self.graph if d.metadata["metadata"].get("parent_id", None) == parent_id]
    
    def find_document_by_tag(self, tag_to_find: str) -> Document:
        return [
            doc for doc in self.graph 
            if any(link.tag == tag_to_find for link in doc.metadata["links"])
        ]
 
    #endregion
    
    #region Graph Operations
    def documents_to_nx(self) -> nx.DiGraph:
        return documents_to_networkx(self.graph)
        

    def plot_graph(self, file_name):
        net = Network(
            notebook=True,
            cdn_resources="in_line",
            bgcolor="#222222",
            font_color="white",
            height="750px",
            width="100%",
        )
        # Convert the NetworkX graph to a PyVis graph
        G = documents_to_networkx(self.graph, tag_nodes=True)

        net.from_nx(G)

        # Render the graph in the Jupyter notebook
        net.show_buttons()
        net.show(file_name + ".html")

        return
    #endregion
