import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class G46HW1 {

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: number_partitions, <path to file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true)
                .setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> dataset = sc.textFile(args[1]).repartition(K);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Long> count;

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // VERSION WITH DETERMINISTIC PARTITIONS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = dataset
                .mapToPair((row) -> {    // <-- MAP PHASE (R1)
                    String[] parts = row.split(" ");
                    // map the pair into the bucket "i mod K"
                    return new Tuple2<>(Long.parseLong(parts[0]) % K, parts[1]);
                })
                .groupByKey()    // <-- REDUCE PHASE (R1)
                .flatMapToPair((tuple) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    // count the repetition of each word
                    for (String c : tuple._2()) {
                        counts.put(c, 1 + counts.getOrDefault(c, 0L));
                    }
                    // convert the dictionary into an ArrayList
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .reduceByKey(Long::sum) // group by word and sum the values
                .sortByKey(); // sort by word, so in alphabetical order
        List<Tuple2<String, Long>> result = count.collect();
        System.out.print("VERSION WITH DETERMINISTIC PARTITIONS\nOutput pairs = ");
        result.forEach(el -> System.out.print(el + " "));

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // VERSION WITH SPARK PARTITIONS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = dataset
                // consider each partition
                .mapPartitionsToPair((row) -> {    // <-- MAP PHASE (R1)
                    HashMap<String, Long> counts = new HashMap<>();
                    long size = 0;
                    // foreach entry in the partition update the dictionary that contains
                    // the count of each word and increase by 1 the variable size that
                    // contains the size of the partition
                    while (row.hasNext()) {
                        String key = row.next().split(" ")[1];
                        counts.put(key, 1 + counts.getOrDefault(key, 0L));
                        size += 1;
                    }
                    // create the special pair for storing the partition size
                    counts.put("partitionSize", size);

                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- REDUCE PHASE (R2)
                .mapToPair((it) -> {
                    // if the key is "partitionSize" take the maximum among the elements
                    if (it._1().equals("partitionSize")) {
                        long max = 0;
                        for (long c : it._2())
                            max = Math.max(c, max);
                        return new Tuple2<>("maxPartitionSize", max);
                    } else {
                        // else (for regular keys) compute the sum of all elements
                        long sum = 0;
                        for (long c : it._2())
                            sum += c;
                        return new Tuple2<>(it._1(), sum);
                    }
                });

        Long N_max = count
                .filter((el) -> el._1().equals("maxPartitionSize")) // get the pair with key "maxPartitionSize"
                .first()
                ._2(); // get the value

        Tuple2<String, Long> tuple = count
                .filter((el) -> !el._1().equals("maxPartitionSize")) // exclude the pair with key "maxPartitionSize"
                .reduce((value1, value2) -> {
                    if (value1._2() > value2._2() || // if the first has a higher count
                            (value1._2().equals(value2._2()) && value1._1().compareTo(value2._1()) < 0))
                        // or It has the an equal count and It comes before in alphabetical order
                        return value1;
                    else
                        return value2;
                });
        // we use reduce function, that scan only once the list (O(n)), to get the maximum among all values
        // because it has lower complexity w.r.t. ordering (O(n*log(n))) and taking the first element

        System.out.println("\n\nVERSION WITH SPARK PARTITIONS\n" +
                "Most frequent class =  " + tuple +
                "\nMax partition size =  " + N_max
        );
    }
}
