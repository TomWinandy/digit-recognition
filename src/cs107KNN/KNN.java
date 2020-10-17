package cs107KNN;

import java.util.Arrays;

public class KNN {
	public static void main(String[] args) {

		int TESTS = 1000;
		int K = 7;

		byte[][][] trainImages = parseIDXimages(Helpers.readBinaryFile("datasets/100-per-digit_images_train"));
		System.out.println("Parsing des images terminé");
		byte[] trainLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/100-per-digit_labels_train"));
		System.out.println("Parsing des labels terminé");

		byte[][][] testImages = parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test"));
		System.out.println("Parsing des images de test terminé");
		byte[] testLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/10k_labels_test"));
		System.out.println("Parsing des labels de test terminé");

		byte[] predictions = new byte[TESTS];
		long start = System.currentTimeMillis();
		for (int i = 0; i < TESTS; i++) {
			predictions[i] = knnClassify(testImages[i], trainImages, trainLabels, K);
		}
		long end = System.currentTimeMillis();
		double time = (end - start) / 1000d;

		System.out
				.println("Accuracy = " + accuracy(predictions, Arrays.copyOfRange(testLabels, 0, TESTS)) * 100 + " %");
		System.out.println("Time = " + time + " seconds");
		System.out.println("Time per test image = " + (time / TESTS));

		Helpers.show("Tests", testImages, predictions, testLabels, (int) Math.sqrt(TESTS), (int) Math.sqrt(TESTS));
	}

	/**
	 * Composes four bytes into an integer using big endian convention.
	 *
	 * @param bXToBY The byte containing the bits to store between positions X and Y
	 * 
	 * @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
	 */
	public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {
		return ((b31ToB24 & 0xFF) << 24) + ((b23ToB16 & 0xFF) << 16) + ((b15ToB8 & 0xFF) << 8) + ((b7ToB0 & 0xFF) << 0);
	}

	/**
	 * Converts a byte from unsigned to signed
	 * 
	 * @param unsigned the byte to convert
	 * 
	 * @return a (converted) byte
	 */
	public static byte unsignedToSigned(byte unsigned) {
		byte signed = (byte) ((unsigned & 0xFF) - 128);
		return signed;
	}

	/**
	 * Parses an IDX file containing images
	 *
	 * @param data the binary content of the file
	 *
	 * @return A tensor of images
	 */
	public static byte[][][] parseIDXimages(byte[] data) {
		assert (data != null);

		// On arrête l'exécution si le nombre magique n'est pas le bon (i.e. 2051)
		int nombreMagique = extractInt(data[0], data[1], data[2], data[3]);
		if (nombreMagique != 2051) {
			return null;
		}

		// On récupère les valeurs qui correspondent aux dimensions du tenseur, et on le
		// déclare
		int nbImages = extractInt(data[4], data[5], data[6], data[7]);
		int hauteur = extractInt(data[8], data[9], data[10], data[11]);
		int largeur = extractInt(data[12], data[13], data[14], data[15]);
		byte[][][] tenseur = new byte[nbImages][hauteur][largeur];

		// On initialise chaque pixel du tenseur en utilisant le tableau fourni (data)
		for (int idImage = 0; idImage < tenseur.length; ++idImage) {
			for (int idLigne = 0; idLigne < tenseur[idImage].length; ++idLigne) {
				for (int idColonne = 0; idColonne < tenseur[idImage][idLigne].length; ++idColonne) {
					tenseur[idImage][idLigne][idColonne] = unsignedToSigned(
							data[16 + (idImage * hauteur * largeur) + (idLigne * largeur) + (idColonne)]);
					/*
					 * La ligne précédente nécessite quelques explications : On veut attribuer à
					 * chaque case du tenseur d'image la valeur du pixel correspondant. On doit
					 * alors accéder à ce pixel : - On commence à exploiter "data" à partir de la
					 * 17ème case, les 16 premières étant utilisées pour les informations que l'on a
					 * récupérées plus haut. - On se décale ensuite dans le tableau du nombre de
					 * pixels qu'il y a dans les images que l'on a déjà enregistrées. - Puis on se
					 * décale du nombre de pixels qu'il y a dans les lignes que l'on a déjà
					 * enregistrées (dans l'image que l'on considère actuellement). - Finalement, on
					 * se décale du nombre de pixels que l'on a déjà enregistrés dans la ligne que
					 * l'on considère actuellement.
					 */
				}
			}
		}

		return tenseur;
	}

	/**
	 * Parses an idx images containing labels
	 *
	 * @param data the binary content of the file
	 *
	 * @return the parsed labels
	 */
	public static byte[] parseIDXlabels(byte[] data) {
		assert (data != null);

		// On arrête l'exécution si le nombre magique n'est pas le bon (i.e. 2049)
		int nombreMagique = extractInt(data[0], data[1], data[2], data[3]);
		if (nombreMagique != 2049) {
			return null;
		}

		// On récupère la valeur qui correspond à la dimension du tableau
		int nbEtiquettes = extractInt(data[4], data[5], data[6], data[7]);
		byte[] etiquettes = new byte[nbEtiquettes];

		// On récupère et on stocke les étiquettes
		for (int idEtiquette = 0; idEtiquette < etiquettes.length; ++idEtiquette) {
			etiquettes[idEtiquette] = data[8 + idEtiquette];
			// On décale le pointeur de 8 car les 8 premières cases de data ont déjà été
			// utilisées plus haut.
		}

		return etiquettes;
	}

	/**
	 * @brief Computes the squared L2 distance of two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the squared euclidean distance between the two images
	 */
	public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {
		assert (a != null & b != null);
		assert (a.length > 0);
		assert (a.length == b.length && a[0].length == b[0].length);

		float distanceAuCarre = 0;

		// On somme le carré de la distance entre chaque pixels de même position des
		// deux images à comparer
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				distanceAuCarre += Math.pow(a[i][j] - b[i][j], 2);
			}
		}

		return distanceAuCarre;
	}

	/**
	 * @brief Computes the inverted similarity between 2 images.
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the inverted similarity between the two images
	 */
	public static float invertedSimilarity(byte[][] a, byte[][] b) {
		assert (a != null && b != null);
		assert (a.length > 0);
		assert (a.length == b.length && a[0].length == b[0].length);

		// On déclare des variables pour simplifier le calcul, en le séparant selon ses
		// différents termes
		float numerateur = 0;
		float denominateurPremierFacteur = 0;
		float denominateurSecondFacteur = 0;

		// On détermine la moyenne de la valeur des pixels de chaque image
		float[] moyennesImages = moyenneDeuxImages(a, b);
		float moyenneA = moyennesImages[0];
		float moyenneB = moyennesImages[1];

		/*
		 * On somme, séparément : - Le produit des différences de la valeur de chaque
		 * pixel d'une image avec la valeur moyenne des pixels de cette image. - Le
		 * carré de la différence des pixels de la première image avec la moyenne des
		 * pixels de cette image - Idem que la ligne précédente, avec la deuxième image
		 */
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				numerateur += (a[i][j] - moyenneA) * (b[i][j] - moyenneB);
				denominateurPremierFacteur += Math.pow(a[i][j] - moyenneA, 2);
				denominateurSecondFacteur += Math.pow(b[i][j] - moyenneB, 2);
			}
		}
		// Finalement on assemble les sommes calculées, pour obtenir la valeur donnée
		// par la formule de la similarité inversée
		float denominateur = (float) Math.sqrt(denominateurPremierFacteur * denominateurSecondFacteur);

		if (denominateur == 0.0) {
			return 2;
		}

		float resultat = 1 - numerateur / denominateur;

		return resultat;
	}

	/**
	 * @briefs Computes the pixels values average of two images
	 * 
	 * @param a, b two images of the same dimension
	 * 
	 * @return a byte array of length two, containing the average of the pixels of
	 *         both a and b
	 */
	private static float[] moyenneDeuxImages(byte[][] a, byte[][] b) {
		assert (a != null && b != null);
		assert (a.length > 0);
		assert (a.length == b.length && a[0].length == b[0].length);

		float sommePixelsA = 0;
		float sommePixelsB = 0;

		// On somme séparément les pixels de chaque image
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				sommePixelsA += a[i][j];
				sommePixelsB += b[i][j];
			}
		}

		// On calcule la moyenne des pixels de chaque image et on les renvoie, dans un
		// tableau
		float[] moyennesImages = new float[2];
		moyennesImages[0] = (float) (Math.pow(a.length * a[0].length, -1) * sommePixelsA);
		moyennesImages[1] = (float) (Math.pow(b.length * b[0].length, -1) * sommePixelsB);
		return moyennesImages;
	}

	/**
	 * @brief Quicksorts and returns the new indices of each value.
	 * 
	 * @param values the values whose indices have to be sorted in non decreasing
	 *               order
	 * 
	 * @return the array of sorted indices
	 * 
	 *         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
	 */
	public static int[] quicksortIndices(float[] values) {
		assert (values != null);

		// On initialise le tableau des indices triés
		int[] indices = new int[values.length];
		for (int i = 0; i < indices.length; ++i) {
			indices[i] = i;
		}

		// On invoque la méthode quicksortIndices a 4 arguments pour trier le tableau
		quicksortIndices(values, indices, 0, values.length - 1);

		return indices;
	}

	/**
	 * @brief Sorts the provided values between two indices while applying the same
	 *        transformations to the array of indices
	 * 
	 * @param values  the values to sort
	 * @param indices the indices to sort according to the corresponding values
	 * @param         low, high are the **inclusive** bounds of the portion of array
	 *                to sort
	 */
	public static void quicksortIndices(float[] values, int[] indices, int low, int high) {
		assert (values != null && indices != null);
		assert (values.length == indices.length);
		assert (high >= low && high < values.length && low >= 0);

		// Initialisation des variables nécessaires a l'exécution de l'algorithme
		int l = low;
		int h = high;
		float pivot = values[low];

		// Exécution de l'algorithme Quicksort
		while (l <= h) {
			if (values[l] < pivot) {
				++l;
			} else if (values[h] > pivot) {
				--h;
			} else {
				swap(l, h, values, indices);
				++l;
				--h;
			}
		}

		// Appel récursif de l'algorithme Quicksort
		if (low < h) {
			quicksortIndices(values, indices, low, h);
		}
		if (high > l) {
			quicksortIndices(values, indices, l, high);
		}
	}

	/**
	 * @brief Swaps the elements of the given arrays at the provided positions
	 * 
	 * @param         i, j the indices of the elements to swap
	 * @param values  the array floats whose values are to be swapped
	 * @param indices the array of ints whose values are to be swapped
	 */
	public static void swap(int i, int j, float[] values, int[] indices) {
		assert (values != null && indices != null);
		assert (values.length == indices.length);
		assert (i < values.length && i >= 0 && j < values.length && j >= 0);

		// Échange les éléments aux positions i, j du tableau values
		float tmp1 = values[i];
		values[i] = values[j];
		values[j] = tmp1;

		// Échange les éléments aux positions i, j du tableau indices
		int tmp2 = indices[i];
		indices[i] = indices[j];
		indices[j] = tmp2;
	}

	/**
	 * @brief Returns the index of the largest element in the array
	 * 
	 * @param array an array of integers
	 * 
	 * @return the index of the largest integer
	 */
	public static int indexOfMax(int[] array) {
		assert (array != null);

		int idMax = 0;

		// On parcourt le tableau a la recherche du maximum
		for (int i = 0; i < array.length; ++i) {
			if (array[i] > array[idMax]) {
				idMax = i;
			}
		}

		return idMax;
	}

	/**
	 * The k first elements of the provided array vote for a label
	 *
	 * @param sortedIndices the indices sorted by non-decreasing distance
	 * @param labels        the labels corresponding to the indices
	 * @param k             the number of labels asked to vote
	 *
	 * @return the winner of the election
	 */
	public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {
		assert (sortedIndices != null && labels != null);
		assert (sortedIndices.length == labels.length);
		assert (k <= labels.length);

		// On initialise le tableau et on comptabilise les votes
		int[] votes = new int[10];
		for (int i = 0; i < k; ++i) {
			++votes[labels[sortedIndices[i]]];
		}

		return (byte) indexOfMax(votes);
	}

	/**
	 * Classifies the symbol drawn on the provided image
	 *
	 * @param image       the image to classify
	 * @param trainImages the tensor of training images
	 * @param trainLabels the list of labels corresponding to the training images
	 * @param k           the number of voters in the election process
	 *
	 * @return the label of the image
	 */
	public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {
		assert (trainImages != null && trainLabels != null && image != null);
		assert (trainImages.length == trainLabels.length);
		assert (k <= trainLabels.length);

		// On initialise et on construit le tableau des distances
		float[] distances = new float[trainImages.length];
		for (int i = 0; i < distances.length; ++i) {
			// distances[i] = squaredEuclideanDistance(image, trainImages[i]);
			distances[i] = invertedSimilarity(image, trainImages[i]);
		}

		// On initialise et on calcule le résultat
		byte resultat = electLabel(quicksortIndices(distances), trainLabels, k);

		return resultat;
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 * 
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 * 
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {
		assert (predictedLabels != null && trueLabels != null);
		assert (predictedLabels.length == trueLabels.length);

		// On initialise et on calcule le nombre de prédictions correctes
		double nbPredictionCorrectes = 0;

		for (int i = 0; i < predictedLabels.length; i++) {
			if (predictedLabels[i] == trueLabels[i]) {
				++nbPredictionCorrectes;
			}
		}

		// On détermine et on renvoie le taux
		return nbPredictionCorrectes / predictedLabels.length;
	}
}
