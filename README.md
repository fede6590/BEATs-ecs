# BEATs for ECS deployment

This development is about making available the Sound Classification feature using BEATs model on AWS ECS (FARGATE).

It takes messages from "SC_SQS_INPUT", which are event notifications by PutObject action (with .wav suffix) from an S3 bucket (the one receiving the .wav files). THe .wav file is download from the bucket, pre-process and process by BEATs model, and the prediction is finally out to "SC_SQS_OUTPUT".

"SC_SQS_INPUT" and "SC_SQS_OUTPUT" are both environment variables pointing to the respective URL queue.