from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, edm

service_endpoint = "https://tuto.search.windows.net"
api_key = "8064116c-cf7d-49ea-8adb-8deb7637596d"

index_client = SearchIndexClient(service_endpoint, AzureKeyCredential(api_key))

fields = [
    SimpleField(name="id", type=edm.String, key=True),
    SimpleField(name="content", type=edm.String, searchable=True),
]

index = SearchIndex(name="sample-index", fields=fields)

index_client.create_index(index)

search_client = SearchClient(service_endpoint, "sample-index", AzureKeyCredential(api_key))

documents = [
    {"id": "1", "content": "Hello world"},
    {"id": "2", "content": "Azure Cognitive Search"}
]

search_client.upload_documents(documents)