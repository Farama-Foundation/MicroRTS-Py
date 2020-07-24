# pip install boto3
import boto3
import re
client = boto3.client('batch')

# print("creating job queue")
# response = client.create_job_queue(
#     jobQueueName='gym-microrts',
#     state='ENABLED',
#     priority=100,
#     computeEnvironmentOrder=[
#         {
#             'order': 100,
#             'computeEnvironment': 'cleanrl'
#         }
#     ]
# )
# print(response)
# print("job queue created \n=============================")

# print("creating job definition")
response = client.register_job_definition(
    jobDefinitionName='gym-microrts',
    type='container',
    containerProperties={
        'image': 'vwxyzjn/gym-microrts:latest',
        'vcpus': 1,
        'memory': 2000,
    },
    retryStrategy={
        'attempts': 3
    },
    timeout={
        'attemptDurationSeconds': 1800
    }
)
print(response)
print("job definition created \n=============================")

