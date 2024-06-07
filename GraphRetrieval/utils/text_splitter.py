from langchain_core.text_splitter import CharacterTextSplitter

class CustomTextSplitter(CharacterTextSplitter):
    def __init__(self, separator="\n\n", chunk_size=1250, chunk_overlap=100, length_function=len, is_separator_regex=False):
        super().__init__(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function, is_separator_regex=is_separator_regex)

    def create_documents(self, texts):
        return super().create_documents(texts)
