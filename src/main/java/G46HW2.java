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

        List<Vector> inputPoints = readVectorsSeq(filename);

        long startTime = System.currentTimeMillis();

        double maxDistance = 0;//exactMPD(inputPoints);

        long estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("EXACT ALGORITHM\n Max distance = " + maxDistance
                + "\nRunning time = " + estimatedTime + " ms"
        );

        startTime = System.currentTimeMillis();

        maxDistance = twoApproxMPD(inputPoints, K, SEED);

        estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("\n\n2-APPROXIMATION ALGORITHM\n k = " + K
                + "\nMax distance = " + maxDistance
                + "\nRunning time = " + estimatedTime + " ms"
        );

        startTime = System.currentTimeMillis();

        maxDistance = exactMPD(kCenterMPD(inputPoints, K, SEED));

        estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("\n\nk-CENTER-BASED ALGORITHM\n k = " + K
                + "\nMax distance = " + maxDistance
                + "\nRunning time = " + estimatedTime + " ms"
        );
    }

    public static double exactMPD(List<Vector> inputPoints) {
        double max_length = 0;
        for (int i = 0; i < inputPoints.size() - 1; i++)
            // j starts from (i+1) because the distance function is symmetric and
            // for each pair x,y we compute only d(x,y) and not d(y,x)
            for (int j = i + 1; j < inputPoints.size(); j++)
                max_length = Math.max(max_length, Vectors.sqdist(inputPoints.get(i), inputPoints.get(j)));
        return max_length;
    }

    public static double twoApproxMPD(List<Vector> inputPoints, int k, long SEED) {
        Random r = new Random();
        r.setSeed(SEED);

        // generate randomly a vector of 'k' distinct element of 'inputPoints'
        List<Vector> smallSet = r.ints(0, inputPoints.size())
                .distinct()
                .limit(k)
                .mapToObj(inputPoints::get)
                .collect(Collectors.toList());
//        The commented version is slightly better (estimated 2%), but less pretty
//        Set<Integer> randomIndexes = new HashSet<>();
//        while(randomIndexes.size()<k)
//            randomIndexes.add(r.nextInt(inputPoints.size()));
//
//        List<Vector> smallSet = new ArrayList<Vector>();
//        randomIndexes.forEach(idx-> smallSet.add(inputPoints.get(idx)));

        double maxDistance = 0;
        for (Vector center : smallSet)
            for (Vector point : inputPoints)
                maxDistance = Math.max(maxDistance, Vectors.sqdist(center, point));
        return maxDistance;
    }

    public static ArrayList<Vector> kCenterMPD(List<Vector> inputPoints, int k, long SEED) {
        Random r = new Random();
        r.setSeed(SEED);

        // the array that will contain the result
        ArrayList<Vector> C = new ArrayList<>(k);

        // the index of the fist point (generated randomly)
        int indexOfLatestPoint = r.nextInt(inputPoints.size());

        C.add(inputPoints.get(indexOfLatestPoint));

        // I use array because I know the exact size and they are much faster than ArrayList
        double[] distances = new double[inputPoints.size()];
        Arrays.fill(distances, Double.MAX_VALUE);

        for (int center = 1; center < k; center++) {
            double maxDistance = 0;
            int indexOfMaxDistance = 0;

            for (int i = 0; i < inputPoints.size(); i++) {
                // Here we update the distance stored considering also the last point added to 'C'
                // calculating the minimum between the old distance and the distance to the last point
                distances[i] = Math.min(
                        distances[i],
                        Vectors.sqdist(inputPoints.get(i), inputPoints.get(indexOfLatestPoint))
                );
                if (distances[i] > maxDistance) {
                    maxDistance = distances[i];
                    indexOfMaxDistance = i;
                }
            }
            C.add(inputPoints.get(indexOfMaxDistance)); // add the point with maximum distance to C
            indexOfLatestPoint = indexOfMaxDistance;    // save the index to be able to recompute new distances
        }
        return C;
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
