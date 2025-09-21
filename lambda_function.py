# lambda_function.py
import awsgi2
from pipeline_server import app

def lambda_handler(event, context):
    """
    AWS Lambda handler that serves the Flask application via awsgi2.
    """
    return awsgi2.response(app, event, context)
