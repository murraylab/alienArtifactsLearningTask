from storages.backends.azure_storage import AzureStorage

class AzureMediaStorage(AzureStorage):
    account_name = 'alienartifactsstorage' # Must be replaced by your <storage_account_name>
    account_key = '' # Must be replaced by your <storage_account_key>
    azure_container = 'media'
    expiration_secs = None

class AzureStaticStorage(AzureStorage):
    account_name = 'alienartifactsstorage' # Must be replaced by your storage_account_name
    account_key = '' # Must be replaced by your <storage_account_key>
    azure_container = 'static'
    expiration_secs = None