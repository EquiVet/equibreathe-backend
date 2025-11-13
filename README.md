To run locally:

conda activate equi-backend

uvicorn app.main:app --reload --port 8080

To test it locally:

curl -X POST "http://127.0.0.1:8080/predict/nostrils" \
  -F "file=@/cfs/home/u021554/videos_test/Imperador_narinas.mp4"

curl -X POST "http://127.0.0.1:8080/predict/abdomen" \
  -F "file=@/cfs/home/u021554/videos_test/Imperador_abdomen.mp4"

curl -X POST "http://127.0.0.1:8080/predict/both" \
  -F "nostrils=@/cfs/home/u021554/videos_test/Imperador_narinas.mp4" \
  -F "abdomen=@/cfs/home/u021554/videos_test/Imperador_abdomen.mp4"



To deploy on GoogleCloudRun:
1. Make sure you're in the right project:

2. Build image and deploy:
gcloud builds submit --tag gcr.io/equibreathe/equi-backend
gcloud run deploy equi-backend --image gcr.io/equibreathe/equi-backend --platform managed --region europe-west4 --allow-unauthenticated
