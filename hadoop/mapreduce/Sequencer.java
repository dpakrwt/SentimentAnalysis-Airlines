import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

class SequenceMapper extends Mapper<LongWritable, Text, SequenceKey, Text> {
	long start_offset = 0;
	long num_records = 0;

	public void setup(Context con) {
		FileSplit split = (FileSplit) con.getInputSplit();
		start_offset = split.getStart();
		num_records = 0;
		//System.out.println(split.getStart() + "--" + split.getLength());
	}

	public void map(LongWritable key, Text value, Context con)
			throws IOException, InterruptedException {
		num_records++;
		LongWritable offset = new LongWritable(start_offset);
		LongWritable row_nm = new LongWritable(num_records);
		con.write(new SequenceKey(offset,row_nm),
				new Text(value.toString() + "\t" + String.valueOf(num_records)));
	}
}

class SequenceReducer extends Reducer<SequenceKey, Text, Text, Text> {
	long last_split_last_rcd_index = 0;
	String data[] = null;
	long index = 0, count = 0;

	public void reduce(SequenceKey key, Iterable<Text> values, Context con)
			throws IOException, InterruptedException {

		System.out.println("PROCESSING KEY => " + key.offset.toString());

		for (Text value : values) {
			data = value.toString().split("\t", -1);
			String record = data[0];
			count = Long.parseLong(data[1]);
			//System.out.println("Count = " + count);
			index = last_split_last_rcd_index + count;

			con.write(new Text(String.valueOf(index)), new Text(record));
		}

		System.out.println("Last processed index = " + index);
		System.out.println("last_split_last_rcd_index = " + last_split_last_rcd_index);
		
		last_split_last_rcd_index = index;
	}
}

class SequenceKey implements WritableComparable<SequenceKey> {
	LongWritable offset, row_num;
	
	public SequenceKey() {
		this.offset = new LongWritable();
		this.row_num = new LongWritable();
	}

	public SequenceKey(LongWritable offset, LongWritable row_num) {
		this.offset = offset;
		this.row_num = row_num;
	}

	public LongWritable getOffset() {
		return offset;
	}

	public void setOffset(LongWritable offset) {
		this.offset = offset;
	}

	public LongWritable getRow_num() {
		return row_num;
	}

	public void setRow_num(LongWritable row_num) {
		this.row_num = row_num;
	}

	public void readFields(DataInput in) throws IOException {
		offset.readFields(in);
		row_num.readFields(in);
	}

	public void write(DataOutput out) throws IOException {
		offset.write(out);
		row_num.write(out);
	}

	public int compareTo(SequenceKey o) {
		if (this.offset.compareTo(o.offset) == 0)
			return this.row_num.compareTo(o.row_num);
		else
			return this.offset.compareTo(o.offset);
	}

	public boolean equals(Object obj) {
		if (obj instanceof SequenceKey) {
			SequenceKey other = (SequenceKey) obj;
			return other.offset.equals(this.offset)
					&& other.row_num.equals(this.row_num);
		}
		return false;
	}
	
	public int hashCode() {
		return this.offset.hashCode();
	}

}

class SequencePartitioner extends Partitioner<SequenceKey, Text> {

	public int getPartition(SequenceKey key, Text value, int numReduceTasks) {
		// TODO Auto-generated method stub
		return new HashPartitioner().getPartition(key.getOffset(), value, numReduceTasks);
	}
	
}

class SequenceGroupComparator extends WritableComparator {
	public SequenceGroupComparator() {
		super(SequenceKey.class, true);
	}
	
	public int compare(WritableComparable w1, WritableComparable w2) {
		SequenceKey k1 = (SequenceKey) w1;
		SequenceKey k2 = (SequenceKey) w2;
		
		return k1.offset.compareTo(k2.offset);
	}
}

class SequenceSortComparator extends WritableComparator {
	public SequenceSortComparator() {
		super(SequenceKey.class, true);
	}
	
	public int compare(WritableComparable w1, WritableComparable w2) {
		SequenceKey k1 = (SequenceKey) w1;
		SequenceKey k2 = (SequenceKey) w2;
		
		if(k1.offset.compareTo(k2.offset) == 0)
			return k1.row_num.compareTo(k2.row_num);
		
		return k1.offset.compareTo(k2.offset);
	}
}

public class Sequencer extends Configured implements Tool {
	public int run(String[] args) throws Exception {
		Configuration conf1 = new Configuration();

		//conf1.setInt("mapreduce.input.fileinputformat.split.maxsize", 500);

		Job job1 = Job.getInstance(conf1);
		job1.setJarByClass(getClass());
		job1.setMapperClass(SequenceMapper.class);
		job1.setNumReduceTasks(1);
		job1.setReducerClass(SequenceReducer.class);
		job1.setMapOutputKeyClass(SequenceKey.class);
		job1.setMapOutputValueClass(Text.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(Text.class);
		job1.setInputFormatClass(TextInputFormat.class);
		job1.setPartitionerClass(SequencePartitioner.class);
		job1.setSortComparatorClass(SequenceSortComparator.class);
		job1.setGroupingComparatorClass(SequenceGroupComparator.class);
		FileInputFormat.addInputPath(job1, new Path(args[0]));
		FileOutputFormat.setOutputPath(job1, new Path(args[1]));

		return job1.waitForCompletion(true) ? 0 : 1;
	}

	public static void main(String args[]) throws Exception {
		int rc = ToolRunner.run(new Configuration(), new Sequencer(), args);
		System.exit(rc);

		/*
		 * JobControl c = new CustomerAnalyse().creatControl(args); //Thread
		 * jobControlThread = new Thread(c); //jobControlThread.start();
		 * c.run(); System.exit(0);
		 */
	}
}