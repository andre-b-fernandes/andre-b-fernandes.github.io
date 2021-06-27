---
layout: home
title: Professional Experience
permalink: /prof-exp/
---


#### (2018 - 2020) Jumpseller - Part-time (Student-worker).

- **Jumpseller** is an **e-commerce provider** where you can easily set up your online store. During this period, I made part of their **software development team**. While working there I coded in **Ruby**, using the known framework **Ruby on Rails** as well as in **ReactJS** for **frontend** development. Besides this, I also worked with **job queues** using **Sidekiq**, **webhooks** for service connection, and with caching mechanisms using **Redis**. In terms of databases, I worked with both **MySQL** and **PostgreSQL** databases. Overall I gained many insights on the development and maintenance of a highly used web application as well as the software development process using **SCRUM**. It was my first job on the field, working part-time and managing my time as a student-worker.

#### (2020 - Today) Jumpseller - Full-time.

- This is the moment I turned myself into a Machine Learning Engineer, and because it was a small company, in terms of the number of employees, I also made a lot of work in **Data Engineering**.
Initially, I was responsible for building a large-scale **Big Data** collection pipeline using AWS services, namely **AWS Kinesis Firehose** for stream-data collection which is stored in **AWS S3**, processed and aggregated using **AWS Glue**, and queried using **AWS Athena**.
I've also helped implement a more robust search mechanism for the main project **ElasticSearch**.

- Currently, I'm working on a large-scale online machine learning project related to recommendations.
I've worked with content-based recommendations using an NLP deep learning neural network architecture called Word2Vec, and also collaborative filtering using item-based recommendations
with cosine similarity and a session-based recommender system. We also use **ElasticSearch** for this task.
This project's architecture is much different than the usual applications I've done so far since it's done mostly via separated **Docker** images stored using **AWS ECR**, and because
it has to interconnect with the main project which has been a challenge but very fun to do. The models are trained and served using **AWS Sagemaker**. It includes a deploy CI/CD pipeline which deploys the project when in the master branch using **Gitlab** pipelines.
I used **Python** and some of its libraries such as **Numpy**, **Tensorflow**, and **Pandas**. I also want to note that a component of this project is an on-demand machine learning model which can be asked for predictions in a **Flask** server.

![Jumpseller](/assets/img/prof-exp/jumpseller.jpg)
