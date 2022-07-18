from storages.backends.azure_storage import AzureStorage

class AzureMediaStorage(AzureStorage):
    account_name = 'alienartifactsstorage' # Must be replaced by your <storage_account_name>
    account_key = 'kJLED723ythCKMTMB59jKdmL8Y9w6HwUM5CFYBVAzJ6cXwEjJLACae7RP52X2UJbHQSrm1h7MbzG2WvVbvpD6w==' # Must be replaced by your <storage_account_key>
    azure_container = 'media'
    expiration_secs = None

class AzureStaticStorage(AzureStorage):
    account_name = 'alienartifactsstorage' # Must be replaced by your storage_account_name
    account_key = 'kJLED723ythCKMTMB59jKdmL8Y9w6HwUM5CFYBVAzJ6cXwEjJLACae7RP52X2UJbHQSrm1h7MbzG2WvVbvpD6w==' # Must be replaced by your <storage_account_key>
    azure_container = 'static'
    expiration_secs = None