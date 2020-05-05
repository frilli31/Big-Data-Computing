import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class G46HW2 {

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Reading points from a file whose name is provided as args[0]
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: file_path k");
        }
        String filename = args[0];
        int K = Integer.parseInt(args[1]);

        final long SEED = 1211142L;

        List<Vector> inputPoints = readVectorsSeq(filename);

        long startTime = System.currentTimeMillis();

        double maxDistance = exactMPD(inputPoints);

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

        ArrayList<Vector> centers = kCenterMPD(inputPoints, K, SEED);
        maxDistance = exactMPD(centers);

        estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("\n\nk-CENTER-BASED ALGORITHM\n k = " + K
                + "\nMax distance = " + maxDistance
                + "\nRunning time = " + estimatedTime + " ms"
        );
    }

    public static double exactMPD(List<Vector> inputPoints) {
        double maxSqDistance = 0;
        for (int i = 0; i < inputPoints.size() - 1; i++)
            // j starts from (i+1) because the distance function is symmetric and
            // for each pair x,y we compute only d(x,y) and not d(y,x)
            for (int j = i + 1; j < inputPoints.size(); j++)
                maxSqDistance = Math.max(maxSqDistance, Vectors.sqdist(inputPoints.get(i), inputPoints.get(j)));
        return Math.sqrt(maxSqDistance);
    }

    public static double twoApproxMPD(List<Vector> inputPoints, int k, long SEED) {
        Random r = new Random();
        r.setSeed(SEED);

        // generate randomly a vector of 'k' element from 'inputPoints' using k distinct indexes
        Set<Integer> randomIndexes = new HashSet<>();
        while (randomIndexes.size() < k)
            randomIndexes.add(r.nextInt(inputPoints.size()));
        List<Vector> sampleSet = new ArrayList<Vector>();
        randomIndexes.forEach(idx -> sampleSet.add(inputPoints.get(idx)));

        double maxSqDistance = 0;
        // test all possible pairs picking one element from the 'sampleSet' and
        // one element from the entire set
        for (Vector sample : sampleSet)
            for (Vector point : inputPoints)
                maxSqDistance = Math.max(maxSqDistance, Vectors.sqdist(sample, point));
        return Math.sqrt(maxSqDistance);
    }

    public static ArrayList<Vector> kCenterMPD(List<Vector> inputPoints, int k, long SEED) {
        Random r = new Random();
        r.setSeed(SEED);

        // the array that will contain the centers that will be returned
        ArrayList<Vector> centers = new ArrayList<>(k);

        // this variable will contain the index of the latest point added to 'centers' which
        // will be used to calculate the new distances from the points to the centers
        int indexOfLatestPoint = r.nextInt(inputPoints.size());

        centers.add(inputPoints.get(indexOfLatestPoint));

        // I use array because I know the exact size and they are faster than ArrayList
        double[] sqDistances = new double[inputPoints.size()];
        Arrays.fill(sqDistances, Double.MAX_VALUE);

        for (int center = 1; center < k; center++) {
            double maxSqDistance = 0;
            int indexOfMaxDistance = 0;

            for (int i = 0; i < inputPoints.size(); i++) {
                // Here we update the distance stored considering also the latest point added to 'centers'
                // calculating the minimum between the old distance and the distance to the point 'i'
                // and the latest point added to 'centers'
                sqDistances[i] = Math.min(
                        sqDistances[i],
                        Vectors.sqdist(inputPoints.get(i), inputPoints.get(indexOfLatestPoint))
                );
                if (sqDistances[i] > maxSqDistance) {
                    maxSqDistance = sqDistances[i];
                    indexOfMaxDistance = i;
                }
            }
            centers.add(inputPoints.get(indexOfMaxDistance)); // add the point with maximum distance to 'centers'
            indexOfLatestPoint = indexOfMaxDistance;    // save the index to be able to recompute new distances
        }
        return centers;
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
