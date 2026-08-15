gcloud functions deploy upload-event-publisher `
    --gen2 `
    --runtime=python310 `
    --region=asia-south1 `
    --source=../cloud-function `
    --entry-point=gcs_trigger `
    --trigger-bucket=photo-micro-upload-test `
    --memory=512Mi `
    --timeout=60s `
    --set-env-vars=PROJECT_ID=test-face-clustering,PUBSUB_TOPIC=photo-upload-events