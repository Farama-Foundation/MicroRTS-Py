VERSION=1.8.0-cuda11.1-cudnn8-runtime-gym-microrts-0.3.3

docker build  -t vwxyzjn/gym-microrts:$VERSION  -t vwxyzjn/gym-microrts:latest -f Dockerfile .

docker push vwxyzjn/gym-microrts:latest
docker push vwxyzjn/gym-microrts:$VERSION
