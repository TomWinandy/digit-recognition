package cs107KNN;

import java.util.Set;
import java.util.HashSet;
import java.util.Random;
import java.util.ArrayList;

public class KMeansClustering {
	public static void main(String[] args) {
		int K = 7;
		int maxIters = 20;

		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/1000-per-digit_images_train"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/1000-per-digit_labels_train"));

		byte[][][] reducedImages = KMeansReduce(images, K, maxIters);

		byte[] reducedLabels = new byte[reducedImages.length];
		for (int i = 0; i < reducedLabels.length; i++) {
			reducedLabels[i] = KNN.knnClassify(reducedImages[i], images, labels, 5);
			System.out.println("Classified " + (i + 1) + " / " + reducedImages.length);
		}

		Helpers.writeBinaryFile("datasets/reduced10Kto1K_images", encodeIDXimages(reducedImages));
		Helpers.writeBinaryFile("datasets/reduced10Kto1K_labels", encodeIDXlabels(reducedLabels));
	}

	/**
	 * Converts a byte from signed to unsigned
	 * 
	 * @param signed, the byte to convert
	 * 
	 * @return a (converted) byte
	 */
	private static byte signedToUnsigned(byte signed) {
		byte unsigned = (byte) ((signed & 0xFF) + 128);
		return unsigned;
	}

	/**
	 * @brief Encodes a tensor of images into an array of data ready to be written
	 *        on a file
	 * 
	 * @param images the tensor of image to encode
	 * 
	 * @return the array of byte ready to be written to an IDX file
	 */
	public static byte[] encodeIDXimages(byte[][][] images) {
		int nbImages = images.length;
		int hauteur = images[0].length;
		int largeur = images[0][0].length;
		int size = 16 + nbImages * hauteur * largeur;
		byte[] data = new byte[size];

		encodeInt(2051, data, 0);
		encodeInt(nbImages, data, 4);
		encodeInt(hauteur, data, 8);
		encodeInt(largeur, data, 12);

		for (int idImage = 0; idImage < nbImages; ++idImage) {
			for (int idLigne = 0; idLigne < hauteur; ++idLigne) {
				for (int idColonne = 0; idColonne < largeur; ++idColonne) {
					data[16 + (idImage * hauteur * largeur) + (idLigne * largeur) + (idColonne)] = signedToUnsigned(
							images[idImage][idLigne][idColonne]);
				}
			}
		}

		return data;
	}

	/**
	 * @brief Prepares the array of labels to be written on a binary file
	 * 
	 * @param labels the array of labels to encode
	 * 
	 * @return the array of bytes ready to be written to an IDX file
	 */
	public static byte[] encodeIDXlabels(byte[] labels) {
		int etiquettes = labels.length;
		int size = 8 + etiquettes;
		byte[] data = new byte[size];

		encodeInt(2049, data, 0);
		encodeInt(etiquettes, data, 4);

		for (int idEtiquette = 0; idEtiquette < etiquettes; idEtiquette++) {
			data[8 + idEtiquette] = labels[idEtiquette];
		}

		return data;
	}

	/**
	 * @brief Decomposes an integer into 4 bytes stored consecutively in the
	 *        destination array starting at position offset
	 * 
	 * @param n           the integer number to encode
	 * @param destination the array where to write the encoded int
	 * @param offset      the position where to store the most significant byte of
	 *                    the integer, the others will follow at offset + 1, offset
	 *                    + 2, offset + 3
	 */
	public static void encodeInt(int n, byte[] destination, int offset) {
		for (int i = 0; i < 4; ++i) {
			destination[offset + i] = (byte) ((n >> (24 - 8 * i)) & 0xFF);
		}
	}

	/**
	 * @brief Runs the KMeans algorithm on the provided tensor to return size
	 *        elements.
	 * 
	 * @param tensor   the tensor of images to reduce
	 * @param size     the number of images in the reduced dataset
	 * @param maxIters the number of iterations of the KMeans algorithm to perform
	 * 
	 * @return the tensor containing the reduced dataset
	 */
	public static byte[][][] KMeansReduce(byte[][][] tensor, int size, int maxIters) {
		int[] assignments = new Random().ints(tensor.length, 0, size).toArray();
		byte[][][] centroids = new byte[size][][];

		initialize(tensor, assignments, centroids);

		int nIter = 0;
		while (nIter < maxIters) {
			// Step 1: Assign points to closest centroid
			recomputeAssignments(tensor, centroids, assignments);
			System.out.println("Recomputed assignments");
			// Step 2: Recompute centroids as average of points
			recomputeCentroids(tensor, centroids, assignments);
			System.out.println("Recomputed centroids");

			System.out.println("KMeans completed iteration " + (nIter + 1) + " / " + maxIters);
			nIter++;
		}

		return centroids;
	}

	/**
	 * @brief Assigns each image to the cluster whose centroid is the closest. It
	 *        modifies.
	 * 
	 * @param tensor      the tensor of images to cluster
	 * @param centroids   the tensor of centroids that represent the cluster of
	 *                    images
	 * @param assignments the vector indicating to what cluster each image belongs
	 *                    to. if j is at position i, then image i belongs to cluster
	 *                    j
	 */
	public static void recomputeAssignments(byte[][][] tensor, byte[][][] centroids, int[] assignments) {
		for (int i = 0; i < tensor.length; ++i) {
			float[] distances = new float[centroids.length];
			for (int j = 0; j < centroids.length; ++j) {
				distances[j] = KNN.squaredEuclideanDistance(tensor[i], centroids[j]);
			}
			assignments[i] = KNN.quicksortIndices(distances)[0];
		}
	}

	/**
	 * @brief Computes the centroid of each cluster by averaging the images in the
	 *        cluster
	 * 
	 * @param tensor      the tensor of images to cluster
	 * @param centroids   the tensor of centroids that represent the cluster of
	 *                    images
	 * @param assignments the vector indicating to what cluster each image belongs
	 *                    to. if j is at position i, then image i belongs to cluster
	 *                    j
	 */
	public static void recomputeCentroids(byte[][][] tensor, byte[][][] centroids, int[] assignments) {
		for (int i = 0; i < centroids.length; ++i) {
			ArrayList<byte[][]> elements = new ArrayList<byte[][]>();
			for (int j = 0; j < tensor.length; j++) {
				if (assignments[j] == i) {
					elements.add(tensor[j]);
				}
			}
			if (elements.size() != 0) {
				centroids[i] = moyenneImages(elements);
			}
			elements.clear();
		}
	}

	private static byte[][] moyenneImages(ArrayList<byte[][]> elements) {
		byte[][] moyenne = new byte[elements.get(0).length][elements.get(0)[0].length];
		for (int i = 0; i < moyenne.length; ++i) {
			for (int j = 0; j < moyenne.length; ++j) {
				float temp = 0;
				for (int n = 0; n < elements.size(); n++) {
					temp += elements.get(n)[i][j];
				}
				moyenne[i][j] = (byte) (temp / elements.size());
			}
		}
		return moyenne;
	}

	/**
	 * Initializes the centroids and assignments for the algorithm. The assignments
	 * are initialized randomly and the centroids are initialized by randomly
	 * choosing images in the tensor.
	 * 
	 * @param tensor      the tensor of images to cluster
	 * @param assignments the vector indicating to what cluster each image belongs
	 *                    to.
	 * @param centroids   the tensor of centroids that represent the cluster of
	 *                    images if j is at position i, then image i belongs to
	 *                    cluster j
	 */
	public static void initialize(byte[][][] tensor, int[] assignments, byte[][][] centroids) {
		Set<Integer> centroidIds = new HashSet<>();
		Random r = new Random("cs107-2018".hashCode());

		while (centroidIds.size() != centroids.length)
			centroidIds.add(r.nextInt(tensor.length));
		Integer[] cids = centroidIds.toArray(new Integer[] {});
		for (int i = 0; i < centroids.length; i++)
			centroids[i] = tensor[cids[i]];
		for (int i = 0; i < assignments.length; i++)
			assignments[i] = cids[r.nextInt(cids.length)];
	}
}
