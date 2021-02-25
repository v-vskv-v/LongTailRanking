import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.*;
import java.util.*;

public class ClickJob extends Configured implements Tool {
    public static class ClickMapper extends Mapper<LongWritable, Text, Text, Text> {
        LinksExtractor linksExtractor;
        QsExtractor qsExtractor;
        QsExtractor_noconv qsExtractor_noconv;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            linksExtractor = new LinksExtractor(context);
            qsExtractor = new QsExtractor(context);
            qsExtractor_noconv = new QsExtractor_noconv(context);
        }
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] arr = value.toString().split("\t");
            String[] links_showed = arr[1].split(",http");
            links_showed[0] =links_showed[0].replaceFirst("http","");
            if (qsExtractor.ids.containsKey(arr[0].split("\\@")[0]) || qsExtractor_noconv.ids.containsKey(arr[0].split("\\@")[0])) {
                int sum_pos;
                double sum_time;
                double sum_shows;
                double sum_clicks;
                double sum_pos_first;
                boolean flag_noClick;
                int q_id = qsExtractor_noconv.ids.containsKey(arr[0].split("\\@")[0]) ? qsExtractor_noconv.ids.get(arr[0].split("\\@")[0]) : qsExtractor.ids.get(arr[0].split("\\@")[0]);
                if (arr.length == 4) {
                    String[] links_clicked = arr[2].split(",http");
                    String[] time_click = arr[3].split(",");
                    links_clicked[0] =links_clicked[0].replaceFirst("http","");

                    for(int i=0; i < links_clicked.length; i++) {
                        links_clicked[i] = links_clicked[i].charAt(links_clicked[i].length()-1)=='/' ? links_clicked[i].substring(0, links_clicked[i].length()-1) : links_clicked[i];
                    }
                    
                    int pos_ = 0;

                    sum_pos = 0;
                    sum_time = 0;
                    sum_shows = links_showed.length; //>= 12 ? 12 : links_showed.length;
                    sum_clicks = links_clicked.length;
                    sum_pos_first = 0;
                    flag_noClick = false;

                    for(String link: links_showed) {
                        link = link.charAt(link.length()-1)=='/' ? link.substring(0, link.length()-1) : link; 
                        int tmp = Arrays.asList(links_clicked).indexOf(link);
                        boolean flag_last;
                        boolean flag_first;
                        Long time_watch;

                        if (tmp == -1) {
                            flag_last = false;
                            flag_first = false;
                            time_watch = 0L;
                        } else {
                            flag_last = (tmp == (links_clicked.length - 1));
                            flag_first = (tmp == 0);
                            if(flag_last) {
                                time_watch = 352L;
                            } else {
                                time_watch = (Long.parseLong(time_click[tmp+1]) - Long.parseLong(time_click[tmp])) / 1000;
                            }
                            sum_pos += pos_+1;
                        }
                        if (flag_first){
                            sum_pos_first += pos_+1;
                        }
                        sum_time += time_watch;
                        pos_ += 1;
                    }
                } else {
                    sum_pos = 0;
                    sum_time = 0;
                    sum_shows = links_showed.length;
                    sum_clicks = 0;
                    sum_pos_first = 0;
                    flag_noClick = true;
                }
                context.write(new Text(""+q_id), new Text(""+sum_pos+"\t"+sum_time+"\t"+sum_shows+"\t"+sum_clicks+"\t"+sum_pos_first+"\t"+flag_noClick));
            }
        }
    }
    public static class ClickReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text query, Iterable<Text> nums, Context context) throws IOException, InterruptedException {
            double sum_clicks = 0.0;
            double sum_shows = 0.0;
            double sum_noClicks = 0.0;
            double Qshow = 0.0;
            double sum_pos =0.0;
            double sum_timeL = 0.0;
            double sum_time = 0.0;
            double sum_pos_first = 0.0;
            for (Text i: nums) {
                Qshow ++;
                String[] tmp = i.toString().split("\t");
                sum_shows += Double.parseDouble(tmp[2]);
                if(tmp[5].equals("true")) {
                    sum_noClicks ++;
                } else {
                    sum_pos += Integer.parseInt(tmp[0]);
                    sum_clicks += Double.parseDouble(tmp[3]);
                    sum_timeL += Math.log1p(Double.parseDouble(tmp[1]));
                    sum_time += Double.parseDouble(tmp[1]);
                    sum_pos_first += Double.parseDouble(tmp[4]);
                }
            }
            sum_time = Math.log1p(sum_time);
            double avg_time = sum_clicks == 0 ? 0 : sum_time / sum_clicks;
            double avg_timeL = sum_clicks == 0 ? 0 : sum_timeL / sum_clicks;
            double avg_pos = sum_clicks == 0 ? 0 : sum_pos / sum_clicks;
            double avg_posF = sum_clicks == 0 ? 0 :  sum_pos_first / (Qshow - sum_noClicks);
            double ClickToShow = sum_clicks / sum_shows;
            double avg_clicks = sum_clicks == 0 ? 0 : sum_clicks / (Qshow - sum_noClicks);
            context.write(query, new Text(""+avg_time+"\t"+avg_timeL+"\t"+avg_pos+"\t"+avg_posF+"\t"+ClickToShow+"\t"+avg_clicks+"\t"+Qshow+"\t"+sum_clicks));
        }
    }
    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(new Configuration());
        job.setJarByClass(ClickJob.class);

        FileInputFormat.setInputPaths(job, new Path(input));
        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setMapperClass(ClickMapper.class);
        job.setJobName("Q");

        FileOutputFormat.setOutputPath(job, new Path(output));
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setReducerClass(ClickReducer.class);
        job.setNumReduceTasks(1);
        
        return job;
    }

    //@Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new ClickJob(), args);
        System.exit(ret);
    }
}