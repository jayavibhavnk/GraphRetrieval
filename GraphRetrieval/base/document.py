from langchain_core.documents.base import Document

class GraphDocument(Document):
    def __init__(self, page_content, metadata):
        super().__init__(page_content=page_content, metadata=metadata)

    def __repr__(self):
        return f"GraphDocument(page_content='{self.page_content}', metadata={self.metadata})"
