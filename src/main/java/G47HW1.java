import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;
import java.util.function.Function;

public class G47HW1 {

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
        JavaRDD<String> docs = sc.textFile(args[1]).repartition(K);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Long> count;

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // VERSION WITH DETERMINISTIC PARTITIONS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = docs
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = document.split("\n");
                    ArrayList<Tuple2<Long, String>> pairs = new ArrayList<>(tokens.length);
                    for (String token : tokens) {
                        String[] parts = token.split(" ");
                        pairs.add(new Tuple2<>(Long.parseLong(parts[0]) % K, parts[1]));
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R2)
                .flatMapToPair((tuple) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    for (String c: tuple._2()) {
                        counts.put(c, 1 + counts.getOrDefault(c, 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                }).sortByKey();
        List<Tuple2<String, Long>> result = count.collect();
//        result.sort(Comparator.comparing(t -> t._1()));
       System.out.print("VERSION WITH DETERMINISTIC PARTITIONS\nOutput pairs = ");
       result.forEach(el -> System.out.print(el+" "));
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // IMPROVED WORD COUNT with mapPartitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        List<Tuple2<String, Long>> result2 = docs
                .mapPartitionsToPair((row) -> {    // <-- MAP PHASE (R1)

                    HashMap<String, Long> counts = new HashMap<>();
                    long length = 0;
                    while (row.hasNext()){
                        String key = row.next().split(" ")[1];
                        counts.put(key, 1 + counts.getOrDefault(key, 0L));
                        length += 1;
                    }
                    counts.put("maxPartitionSize", length);

                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- REDUCE PHASE (R2)
                .mapToPair((it) -> {
                    if(it._1().equals("maxPartitionSize")) {
                        long max = 0;
                        for (long c : it._2()) {
                            if (c>max) {
                                max = c;
                            }
                        }
                        return new Tuple2<>(it._1(), max);
                    } else {
                        long sum = 0;
                        for (long c : it._2()) {
                            sum += c;
                        }
                        return new Tuple2<>(it._1(), sum);
                    }
                })
                .sortByKey()
                .map((it) -> it)
                .sortBy((pair) -> pair._2(), false, K)
                .collect();


        Long N_max = result2.stream().filter((el) -> el._1().equals("maxPartitionSize")).findAny().get()._2;
        System.out.println("\n\nVERSION WITH SPARK PARTITIONS\n" +
                "Most frequent class =  " + result2.get(0) +
                "\nMax partition size =  " + N_max
        );
    }
}
