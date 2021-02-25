import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.Arrays;

public class sDBNJob extends Configured implements Tool {
    public static final String URL = "URLS";
    public static final String QURL = "QURLS";
    public static final String HOST = "HOST";
    public static final String QHOST = "QHOST";

    public static class sDBNMapper extends Mapper<LongWritable, Text, Text, Text> {
        LinksExtractor linksExtractor;
        HostsExtractor hostsExtractor;
        QsExtractor qsExtractor;
        QsExtractor_noconv qsExtractor_noconv;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            linksExtractor = new LinksExtractor(context);
            hostsExtractor = new HostsExtractor(context);
            qsExtractor = new QsExtractor(context);
            qsExtractor_noconv = new QsExtractor_noconv(context);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] arr = value.toString().split("\t");
            String[] links_showed = arr[1].split(",http");
            links_showed[0] =links_showed[0].replaceFirst("http","");

            String[] hosts_showed = new String[links_showed.length];
            for(int i = 0; i < links_showed.length; i++) {
                links_showed[i] = links_showed[i].charAt(links_showed[i].length()-1)=='/' ? links_showed[i].substring(0, links_showed[i].length()-1) : links_showed[i]; 
                hosts_showed[i] = links_showed[i].split("/").length < 3 ? links_showed[i] : links_showed[i].split("/")[2];
                hosts_showed[i] = hosts_showed[i].startsWith("www.") ? hosts_showed[i].replaceFirst("www.", "") : hosts_showed[i];
            }
            
            if (arr.length == 4) {
                String[] links_clicked = arr[2].split(",http");
                links_clicked[0] =links_clicked[0].replaceFirst("http","");

                String[] hosts_clicked = new String[links_clicked.length];
                for(int i = 0; i < links_clicked.length; i++) {
                    links_clicked[i] = links_clicked[i].charAt(links_clicked[i].length()-1)=='/' ? links_clicked[i].substring(0, links_clicked[i].length()-1) : links_clicked[i];
                    hosts_clicked[i] = links_clicked[i].split("/").length < 3 ? links_clicked[i] : links_clicked[i].split("/")[2];
                    hosts_clicked[i] = hosts_clicked[i].startsWith("www.") ? hosts_clicked[i].replaceFirst("www.", "") : hosts_clicked[i];
                }

                String last_clicked_link = links_clicked[links_clicked.length-1];
                int lastclick_pos = Arrays.asList(links_showed).indexOf(last_clicked_link);

                for(int i = 0; i < lastclick_pos+1; i++) {
                    int a_D = 1;
                    int a_N = 0;
                    int s_D = 0;
                    int s_N = i == lastclick_pos ? 1 : 0;
                    int tmp = Arrays.asList(links_clicked).indexOf(links_showed[i]);
                    if (tmp != -1){
                        a_N ++;
                        s_D ++;
                    }
                    if(linksExtractor.ids.containsKey(links_showed[i])) {
                        context.write(new Text(URL + "|" + linksExtractor.ids.get(links_showed[i])), new Text("" + a_D + "\t" + a_N + "\t" + s_D + "\t" + s_N));
                    }
                    if(hostsExtractor.ids.containsKey(hosts_showed[i])) {
                        context.write(new Text(HOST + "|" + hostsExtractor.ids.get(hosts_showed[i])), new Text("" + a_D + "\t" + a_N + "\t" + s_D + "\t" + s_N));
                    }
                    if (qsExtractor.ids.containsKey(arr[0].split("\\@")[0]) || qsExtractor_noconv.ids.containsKey(arr[0].split("\\@")[0])) {
                        int q_id = qsExtractor_noconv.ids.containsKey(arr[0].split("\\@")[0]) ? qsExtractor_noconv.ids.get(arr[0].split("\\@")[0]) : qsExtractor.ids.get(arr[0].split("\\@")[0]);
                        if(linksExtractor.ids.containsKey(links_showed[i])) {
                            context.write(new Text("" + QURL + "|" + q_id + "\t"+ linksExtractor.ids.get(links_showed[i])), new Text("" + a_D + "\t" + a_N + "\t" + s_D + "\t" + s_N));
                        }
                        if(hostsExtractor.ids.containsKey(hosts_showed[i])) {
                            context.write(new Text("" + QHOST + "|" + q_id + "\t" + hostsExtractor.ids.get(hosts_showed[i])), new Text("" + a_D + "\t" + a_N + "\t" + s_D + "\t" + s_N));
                        }
                    }
                }
            }
        }
    }

    public static class sDBNReducer extends Reducer<Text, Text, Text, Text> {

        private MultipleOutputs<Text, Text> multipleOutputs;

        public void setup(final Reducer.Context context) {
            multipleOutputs = new MultipleOutputs(context);

        }

        @Override
        protected void reduce(Text key, Iterable<Text> nums, Context context) throws IOException, InterruptedException {
            String[] keys = key.toString().split("\\|");
            double a_D = 0.0;
            double a_N = 0.0;
            double s_D = 0.0;
            double s_N = 0.0;
            for(Text val : nums){
                String[] tmp = val.toString().split("\t");
                a_D += Double.valueOf(tmp[0]);
                a_N += Double.valueOf(tmp[1]);
                s_D += Double.valueOf(tmp[2]);
                s_N += Double.valueOf(tmp[3]);
            }
            double a = (a_N + 0.1) / (a_D + 0.1 + 0.1);
            double s = (s_N + 0.1) / (s_D + 0.1 + 0.1);
            double r = a * s;
            multipleOutputs.write(new Text(keys[1]), new Text(""+a+"\t"+s+"\t"+r), keys[0]+"/part");
        }
    }

    private Job getJobConf(String input, String output) throws IOException {
        final Job job = Job.getInstance(getConf());

        job.setJarByClass(sDBNJob.class);
        job.setInputFormatClass(TextInputFormat.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputValueClass(Text.class);
        job.setMapOutputKeyClass(Text.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));
        MultipleOutputs.addNamedOutput(job, URL, TextOutputFormat.class, Text.class, Text.class);
        MultipleOutputs.addNamedOutput(job, QURL, TextOutputFormat.class, Text.class, Text.class);
        MultipleOutputs.addNamedOutput(job, HOST, TextOutputFormat.class, Text.class, Text.class);
        MultipleOutputs.addNamedOutput(job, QHOST, TextOutputFormat.class, Text.class, Text.class);

        job.setMapperClass(sDBNMapper.class);
        job.setReducerClass(sDBNReducer.class);
        job.setNumReduceTasks(1);
        return job;
    }

    //@Override
    public int run(String[] args) throws Exception {
        final Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new sDBNJob(), args);
        System.exit(ret);
    }
}