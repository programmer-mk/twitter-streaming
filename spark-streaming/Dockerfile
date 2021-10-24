FROM bde2020/spark-submit:2.4.5-hadoop2.7

LABEL maintainer="Milisav Kovacevic <milisavkovacevic.kovacevic@gmail.com>"

ARG SBT_VERSION
ENV SBT_VERSION=${SBT_VERSION:-1.3.13}

RUN wget -O - https://github.com/sbt/sbt/releases/download/v${SBT_VERSION}/sbt-${SBT_VERSION}.tgz | gunzip | tar -x -C /usr/local

ENV PATH /usr/local/sbt/bin:${PATH}

WORKDIR /app

# Pre-install base libraries
ADD build.sbt /app/
ADD plugins.sbt /app/project/
ADD /src /app/src/
#RUN sbt update

COPY template.sh /
COPY test-script.sh /test-script.sh
COPY instructions.txt /app/

ENV SPARK_APPLICATION_MAIN_CLASS TweetPolarityAggregator
ENV AWS_ACCESS_KEY_ID xxx
ENV AWS_SECRET_ACCESS_KEY xxx


# Copy the build.sbt first, for separate dependency resolving and downloading
ONBUILD COPY build.sbt /app/
ONBUILD COPY project /app/project
ONBUILD RUN sbt update

# Copy the source code and build the application
ONBUILD COPY . /app
RUN cd /app
RUN sbt clean assembly

RUN cd ..

RUN ["chmod", "+x", "/template.sh"]
RUN ["chmod", "+x", "/test-script.sh"]
CMD ["/template.sh"]
#ENTRYPOINT ["/test-script.sh"]