# To build this container, go to ESMValTool root folder and execute:
# docker build -t esmvaltool:latest . -f docker/Dockerfile
FROM continuumio/miniconda3

RUN apt install julia -y && apt-get clean
COPY ./environment.yml /src/ESMValTool/environment.yml
WORKDIR /src/ESMValTool
RUN conda update -y conda pip && conda env update --name base --file environment.yml && conda clean --all -y

COPY ./esmvaltool/install/R /src/ESMValTool/esmvaltool/install/R
RUN Rscript ./esmvaltool/install/R/setup.R

COPY . /src/ESMValTool
RUN pip install -e . && pip cache purge && esmvaltool install Julia

ENTRYPOINT ["esmvaltool"]
