import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class G46HW3 {
    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true)
                .setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path k L");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long startTime = System.currentTimeMillis();

        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt(args[2]);

        JavaRDD<Vector> inputPoints = sc.textFile(filename)
                .map((line)->strToVector(line))
                .repartition(L).cache();

        inputPoints.count(); //to force computation

        long estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("Number of points = " + inputPoints.count()
                + "\nk = " + k
                + "\nL = " + L
                + "\nInitialization time = " + estimatedTime + " ms\n");

        ArrayList<Vector> solution = runMapReduce(inputPoints,k,L);

        double avg = measure(solution);
        System.out.println("Average distance = " + avg);
    }

    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L) {
        final long SEED = 1211142L;

        // Round 1
        long startTime = System.currentTimeMillis();

        JavaRDD<Vector> tmp = pointsRDD.mapPartitions(
                (partition)->wrapper_kCenter(partition,k,SEED)
        );

        tmp.count(); // to force computation

        long estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("Running time of Round 1 = " + estimatedTime + " ms");

        // Round 2
        startTime = System.currentTimeMillis();

        ArrayList<Vector> coreset = new ArrayList(tmp.collect()); //gather all K*L points from RDD
        ArrayList<Vector> result = runSequential(coreset, k);

        estimatedTime = System.currentTimeMillis() - startTime;

        System.out.println("Running time of Round 2 = " + estimatedTime + " ms\n");

        return result;
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Auxiliary methods
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double measure(ArrayList<Vector> pointSet) {
        double sum = 0;
        int numberOfSummedDistances = pointSet.size() * (pointSet.size()-1) / 2;
        for (int i = 0; i < pointSet.size() ; i++)
            for (int j = i+1; j < pointSet.size(); j++)
                sum += java.lang.Math.sqrt(Vectors.sqdist(pointSet.get(i), pointSet.get(j)));

        return sum / numberOfSummedDistances;
    }

    //wrapping method to avoid kCenter signature change
    public static Iterator<Vector> wrapper_kCenter(Iterator<Vector> inputPoints, int k, long SEED){
        // Convert iterator to iterable
        Iterable<Vector> iterable = () -> inputPoints;

        // Create a List from the Iterable
        List<Vector> list = StreamSupport
                .stream(iterable.spliterator(), false)
                .collect(Collectors.toList());
        return kCenter(list,k,SEED).iterator();
    }

    public static ArrayList<Vector> kCenter(List<Vector> inputPoints, int k, long SEED) {
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

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential
}
