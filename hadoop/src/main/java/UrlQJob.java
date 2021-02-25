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

public class UrlQJob extends Configured implements Tool {
    public static class UrlQMapper extends Mapper<LongWritable, Text, Text, Text> {
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
                int q_id = qsExtractor_noconv.ids.containsKey(arr[0].split("\\@")[0]) ? qsExtractor_noconv.ids.get(arr[0].split("\\@")[0]) : qsExtractor.ids.get(arr[0].split("\\@")[0]);
                if (arr.length == 4) {
                    String[] links_clicked = arr[2].split(",http");
                    String[] time_click = arr[3].split(",");
                    links_clicked[0] =links_clicked[0].replaceFirst("http","");

                    for(int i=0; i < links_clicked.length; i++) {
                        links_clicked[i] = links_clicked[i].charAt(links_clicked[i].length()-1)=='/' ? links_clicked[i].substring(0, links_clicked[i].length()-1) : links_clicked[i];
                    }
                    
                    int pos_ = 0;
                    for(String link: links_showed) {
                        if(pos_ == 12) {
                            continue;
                        }
                        link = link.charAt(link.length()-1)=='/' ? link.substring(0, link.length()-1) : link; 
                        if(!linksExtractor.ids.containsKey(link)) {
                            pos_ += 1;
                            continue;
                        }
                        int tmp = Arrays.asList(links_clicked).indexOf(link);
                        boolean flag_last;
                        boolean flag_first;
                        Long time_watch;

                        int onlyClickflag = (links_clicked.length == 1 && tmp != -1) ? 1 : 0;

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
                        }
                        context.write(new Text(""+q_id+"\t"+linksExtractor.ids.get(link)),new Text(""+pos_+"\t"+tmp+"\t"+flag_first+"\t"+flag_last+"\t"+time_watch+"\t"+onlyClickflag+"\t"+links_clicked.length));
                        pos_ += 1;
                    }
                } else {
                    int pos_ = 0;
                    int tmp = -1;
                    boolean flag_last = false;
                    boolean flag_first = false;
                    int time_watch = 0;
                    int onlyClickflag = 0;
                    for(String link: links_showed) {
                        if(pos_ == 12) {
                            continue;
                        }
                        link = link.charAt(link.length()-1)=='/' ? link.substring(0, link.length()-1) : link; //
                        if(!linksExtractor.ids.containsKey(link)) {
                            pos_ += 1;
                            continue;
                        }
                        context.write(new Text(""+q_id+"\t"+linksExtractor.ids.get(link)), new Text(""+pos_+"\t"+tmp+"\t"+flag_first+"\t"+flag_last+"\t"+time_watch+"\t"+onlyClickflag+"\t"+0));
                        pos_ += 1;
                    }
                }
            }
        }
    }

    public static class UrlQReducer extends Reducer<Text, Text, Text, Text> {
        Double[] coef = new Double[]{0.41,0.16,0.105,0.08,0.06,0.05,0.035,0.03,0.025,0.02};
        @Override
        protected void reduce(Text link, Iterable<Text> nums, Context context) throws IOException, InterruptedException {
            double CTR_show = 0.0;
            double CTR_click = 0.0;
            double CTRfirst_click = 0.0;
            double CTRlast_click = 0.0;
            double Meanpos_sum = 0.0;
            double Meanpos_click_sum = 0.0;
            double Meantime_sum = 0.0;
            double[] CTR2pos = new double[12];

            double probLastClick = 0.0;
            double probOnlyClick = 0.0;
            double ClicksBefore_sum = 0.0;
            double ClicksAfter_sum = 0.0;

            double Meanpos_ifClick = 0.0;
            double ClickNotFirst = 0.0;
            double[] CTR2Cpos = new double[10];
            double CTR5_show = 0.0;
            double CTR5_click = 0.0;
            double AvgTime = 0.0;
            double PBM_show = 0.0;
            double PBM_click = 0.0;

            for (Text i: nums) {
                String[] tmp = i.toString().split("\t");
                int pos_ = Integer.parseInt(tmp[0]);
                Meanpos_sum += pos_ + 1;
                if (pos_ < 5) {
                    CTR5_show += 1;
                }
                if (pos_ < 10) {
                    PBM_show += coef[pos_];
                }
                if(tmp[1].equals("-1")) {
                    CTR_show += 1;
            
                } else {
                    CTR_show += 1;
                    CTR_click += 1;
                    CTRfirst_click += (tmp[2] == "true" ? 1:0);
                    CTRlast_click += (tmp[3] == "true" ? 1:0);
                    Meanpos_click_sum += Double.parseDouble(tmp[1])+1;
                    Meantime_sum += Math.log1p(Double.parseDouble(tmp[4]));
                    CTR2pos[pos_] += 1;
                    ClicksBefore_sum += Double.parseDouble(tmp[1]);
                    ClicksAfter_sum += (Double.parseDouble(tmp[6]) - Double.parseDouble(tmp[1]) - 1);
                    probOnlyClick += Double.parseDouble(tmp[5]);

                    Meanpos_ifClick += pos_ + 1;
                    if(Double.parseDouble(tmp[1]) != 0) {
                        ClickNotFirst += 1;
                    }
                    if(Double.parseDouble(tmp[1]) < 10) {
                        CTR2Cpos[Integer.parseInt(tmp[1])] += 1;
                    }
                    if(pos_ < 5) {
                        CTR5_click += 1;
                    }
                    AvgTime += Double.parseDouble(tmp[4]);
                    if(pos_ < 10) {
                        PBM_click += 1;
                    }
                }
            }
            double CTR = CTR_click / CTR_show;
            double CTRf = CTRfirst_click / CTR_show;
            double CTRl = CTRlast_click / CTR_show;
            double Mpos = Meanpos_sum / CTR_show;
            double MCpos = 0;
            double Mtime = Meantime_sum / CTR_show;
            double avgClicksBefore = 0.0;
            double avgClicksAfter = 0.0;
            double CTRNotFirst = ClickNotFirst / CTR_show;
            double CTR5 = CTR5_show==0 ? 0 : CTR5_click / CTR5_show;
            double PBM = PBM_click / PBM_show;
            if (CTR_click != 0.0) {
                MCpos = Meanpos_click_sum / CTR_click;
                probLastClick = CTRl / CTR_click;
                probOnlyClick = probOnlyClick / CTR_click;
                avgClicksBefore = ClicksBefore_sum / CTR_click;
                avgClicksAfter = ClicksAfter_sum / CTR_click;
                Meanpos_ifClick = Meanpos_ifClick / CTR_click;
                for(int j=0; j<10; j++) {
                    CTR2Cpos[j] = CTR2Cpos[j] / CTR_click;
                }
                AvgTime = AvgTime / CTR_click;
            }
            String s = "";
            for (int j = 0; j < 12; j++) {
                CTR2pos[j] = CTR2pos[j] / CTR_show;
                s += ("\t" + CTR2pos[j]);
            }
            String s_ = "";
            for (int j=0; j<10; j++) {
                s_ +=("\t"+CTR2Cpos[j]);
            }
            context.write(link, new Text(""+CTR+"\t"+CTRf+"\t"+CTRl+"\t"+Mpos+"\t"+MCpos+"\t"+Mtime+s+"\t"+probOnlyClick+"\t" + probLastClick+"\t"+avgClicksBefore+"\t"+avgClicksAfter+"\t"+Math.log1p(CTR_show)+"\t"+Math.log1p(CTR_click)
            +"\t"+CTRNotFirst+"\t"+CTR5+"\t"+PBM+"\t"+Meanpos_ifClick+"\t"+AvgTime+s_));
        }
    }

    private Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(new Configuration());
        job.setJarByClass(UrlQJob.class);

        FileInputFormat.setInputPaths(job, new Path(input));
        job.setInputFormatClass(TextInputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setMapperClass(UrlQMapper.class);
        job.setJobName("UQ");
        FileOutputFormat.setOutputPath(job, new Path(output));
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setReducerClass(UrlQReducer.class);
        job.setNumReduceTasks(11);
        
        return job;
    }

    //@Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new UrlQJob(), args);
        System.exit(ret);
    }
}