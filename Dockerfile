FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ADD src .

# Set permissions and specify the command to run
RUN chmod +x /start-gpu.sh
CMD /start-gpu.sh
