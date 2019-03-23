from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse


def kmeans_clustering(posts_file, clusters_amount, tags_in_category_amount):
    with open(posts_file, encoding="utf-8") as inp:
        posts = inp.readlines()

    vectorizer = TfidfVectorizer(max_features=3)
    posts_coordinates = vectorizer.fit_transform(posts)

    print("Features:")
    for i in range(len(vectorizer.get_feature_names())):
        print("feature %d: %s" % (i, vectorizer.get_feature_names()[i]))

    print("\nPosts coordinates:")
    print(posts_coordinates)
    print("\nClustering posts with %d clusters" % clusters_amount)

    model = KMeans(
        n_clusters=clusters_amount,
        init='k-means++',
        max_iter=5,
        n_init=1,
        verbose=True
    )
    groups = model.fit_predict(posts_coordinates)

    print("\nAssigning groups to posts:")
    for idx, group_id in enumerate(groups):
        print("post %d:%d" % (idx, group_id))

    print("\nCentroids coordinates:")
    for idx, centroid in enumerate(model.cluster_centers_):
        print("centroid %d: %s" % (idx, centroid))

    ordered_centroids = model.cluster_centers_.argsort()[:, ::-1]
    tags = vectorizer.get_feature_names()

    print("\nClosest tags for centroids:")
    for idx, centroids in enumerate(ordered_centroids):
        print("Centroid %s:" % idx)
        for centroid_tag in centroids[:tags_in_category_amount]:
            print("#%s" % tags[centroid_tag])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--clusters", help="Number of clusters", type=int, required=False, default=2)
    parser.add_argument("-T", "--tags", help="Tags in category", type=int, required=False, default=2)
    parser.add_argument("-F", "--file", help="File with posts", required=False, default="posts.txt")
    args = parser.parse_args()
    kmeans_clustering(args.file, args.clusters, args.tags)
