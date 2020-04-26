import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class G46HW2 {

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Reading points from a file whose name is provided as args[0]
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        String filename = args[0];
        int K = Integer.parseInt(args[1]);

        final long SEED = 1211142L;

        ArrayList<Vector> inputPoints = readVectorsSeq(filename);

        long startTime = System.currentTimeMillis();

        double maxDistance = exactMPD(inputPoints);

        long estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("EXACT ALGORITHM\n Max distance = " + maxDistance
                + "\nRunning time = " + estimatedTime % 1000 + " ms"
        );

        startTime = System.currentTimeMillis();

        maxDistance = twoApproxMPD(inputPoints, K, SEED);

        estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("\n\n2-APPROXIMATION ALGORITHM\n k = " + K
                + "\nMax distance = " + maxDistance
                + "\nRunning time = " + estimatedTime % 1000 + " ms"
        );

        startTime = System.currentTimeMillis();

        maxDistance = exactMPD(kCenterMPD(inputPoints, K, SEED));

        estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("\n\nk-CENTER-BASED ALGORITHM\n k = " + K
                + "\nMax distance = " + maxDistance
                + "\nRunning time = " + estimatedTime % 1000 + " ms"
        );
    }

    public static double exactMPD(ArrayList<Vector> inputPoints) {
        double max_length = 0;
        int size_minus_1 = inputPoints.size() - 1;
        int size = inputPoints.size();
        for (int i = 0; i < size_minus_1; i++)
            for (int j = i + 1; j < size; j++)
                max_length = Math.max(max_length, Vectors.sqdist(inputPoints.get(i), inputPoints.get(j)));
        return max_length;
    }

    public static double twoApproxMPD(ArrayList<Vector> inputPoints, int k, long SEED) {
        Random r = new Random();
        r.setSeed(SEED);

        List<Vector> smallSet = r.ints(0, inputPoints.size())
                .distinct()
                .limit(k)
                .mapToObj(inputPoints::get)
                .collect(Collectors.toList());

        double maxDistance = 0;
        for (Vector center : smallSet)
            for (Vector point : inputPoints)
                maxDistance = Math.max(maxDistance, Vectors.sqdist(center, point));
        return maxDistance;
    }

    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> inputPoints, int k, long SEED) {
        Random r = new Random();
        r.setSeed(SEED);

        // the array that will contain the result
        ArrayList<Vector> resultC = new ArrayList<>(k);

        // the index of the fist point (generated randomly)
        int indexOfLatestPoint = r.nextInt(inputPoints.size());

        resultC.add(inputPoints.get(indexOfLatestPoint));

        // I use array because I know the exact size and they are much faster than ArrayList
        double[] distances = new double[inputPoints.size()];
        Arrays.fill(distances, Double.MAX_VALUE);

        for (int center = 1; center < k; center++) {
            double maxDistance = 0;
            int indexOfMaxDistance = 0;

            for (int i = 0; i < inputPoints.size(); i++) {
                distances[i] = Math.min(
                        distances[i],
                        Vectors.sqdist(inputPoints.get(i), inputPoints.get(indexOfLatestPoint))
                );
                if (distances[i] > maxDistance) {
                    maxDistance = distances[i];
                    indexOfMaxDistance = i;
                }
            }

            indexOfLatestPoint = indexOfMaxDistance;
            resultC.add(inputPoints.get(indexOfMaxDistance));
        }
        return resultC;
    }


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Auxiliary methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }
}
