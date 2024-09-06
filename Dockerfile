# Use Nvidia CUDA base image
FROM dongjiandocker/sd-api:v3.0.1

WORKDIR /

RUN cd / && \
    wget https://f005.backblazeb2.com/file/demo-image/vast.zip -P / && \
    unzip /vast.zip -d / && rm /vast.zip && \
    rm -rf /vast && \
    rm -rf /openpose && \
    cp -fr /vast/win_server.py /win_server.py && \
    cp -fr /vast/rp_handler.py /rp_handler.py && \
    cp -fr /vast/sam-server.py /sam-server.py && \
    cp -fr /vast/openpose /openpose && \
    cp -fr /vast/utils.py /utils.py && \
    cp -fr /vast/sd_input_template.txt /sd_input_template.txt && \
    cp -fr /vast/start.sh /start.sh


# Add the start and the handler

RUN chmod +x /start.sh
CMD /start.sh