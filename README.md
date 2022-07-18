# Alien Artifacts Learning Task

By Warren Woodrich Pettine, July 2022

This code runs the tasks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

When collecting data for that paper, the task webapp and associated database were hosted on Microsoft Azure. If you prefer to host on another service, the structure of several components will likely need to be modified. 

Provided everything is installed correctly, the app should be able to run locally. When you are considering hosting it remote, make sure to update the Google recaptcha variables `GOOGLE_RECAPTCHA_SITE_KEY` and `GOOGLE_RECAPTCHA_SECRET_KEY`. 
 

If you have questions about the implementation, please contact Warren at warren.pettine@gmail.com.

## Django and PostgreSQL sample for Azure App Service

The code is based on a  is a sample application available via the tutorial at: 
[Build a Python and PostgreSQL web app in Azure App Service](https://docs.microsoft.com/azure/app-service/containers/tutorial-python-postgresql-app). 

The sample is a simple Python Django application that connects to a PostgreSQL database.

The database connection information is specified via environment variables `DBHOST`, `DBPASS`, `DBUSER`, and `DBNAME`. This app always uses the default PostgreSQL port.
The `DBHOST` environment variable is expected to contain *only* the server name, not the full URL, which is constructed at run time (see azuresite/production.py). Similarly, `DBUSER` is expected to contain only the user name, not username@servername as before, because using the simpler `DBHOST` the code can also construct the correct login form at run time (again in azuresite/production.py), avoiding failures that arise when `DBUSER` lacks the @servername portion.  

# Contributing

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
