import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.JobContext;

class QsExtractor {
    Map<String, Integer> ids;
    Integer current_id = -1;

    QsExtractor(JobContext context) {
        ids = get_ids(context);
    }

    private Map<String, Integer> get_ids(JobContext context) {
        Map<String, Integer> map = new HashMap<>();

        try {
            //Path pt=new Path("C:/Users/CENSORED/.../data/queries_b_proc.txt")
            Path pt = new Path("/user/vasiliy.viskov/data/queries_b_proc.txt");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader bufferedReader =new BufferedReader(new InputStreamReader(fs.open(pt), "UTF-8"));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] split = line.split("\t");
                map.put(split[1], Integer.parseInt(split[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }
}

class QsExtractor_noconv {
    Map<String, Integer> ids;
    Integer current_id = -1;

    QsExtractor_noconv(JobContext context) {
        ids = get_ids(context);
    }

    private Map<String, Integer> get_ids(JobContext context) {
        Map<String, Integer> map = new HashMap<>();

        try {
            //Path pt=new Path("C:/Users/CENSORED/.../data/queries.tsv")
            Path pt = new Path("/user/vasiliy.viskov/data/queries.tsv");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader bufferedReader =new BufferedReader(new InputStreamReader(fs.open(pt), "UTF-8"));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] split = line.split("\t");
                map.put(split[1], Integer.parseInt(split[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }
}

class HostsExtractor {
    Map<String, Integer> ids;
    Integer current_id = -1;

    HostsExtractor(JobContext context) {
        ids = get_ids(context);
    }

    private Map<String, Integer> get_ids(JobContext context) {
        Map<String, Integer> map = new HashMap<>();

        try {
            //Path pt=new Path("C:/Users/CENSORED/.../data/host.data")
            Path pt = new Path("/user/vasiliy.viskov/data/host.data");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader bufferedReader =new BufferedReader(new InputStreamReader(fs.open(pt)));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] split = line.split("\t");
                String s = split[1].charAt(split[1].length()-1)=='/' ? split[1].substring(0, split[1].length()-1) : split[1];
                map.put("" + s, Integer.parseInt(split[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }
}

class LinksExtractor {
    Map<String, Integer> ids;

    LinksExtractor(JobContext context) {
        ids = get_ids(context);
    }

    private Map<String, Integer> get_ids(JobContext context) {
        Map<String, Integer> map = new HashMap<>();

        try {
            //Path pt=new Path("C:/Users/CENSORED/.../data/url.data")
            Path pt = new Path("/user/vasiliy.viskov/data/url.data");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader bufferedReader =new BufferedReader(new InputStreamReader(fs.open(pt)));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] split = line.split("\t");
                String s = split[1].charAt(split[1].length()-1)=='/' ? split[1].substring(0, split[1].length()-1) : split[1];
                map.put("://" + s, Integer.parseInt(split[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }
}
