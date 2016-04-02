package edu.drexel.sentiment;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Properties;

public class HDPConfig {
	public double gamma;    // HDP
	public double alpha;    // HDP
	public double beta;     // word
	public double eta;      // author preference
	public double lamda;    // document sentiment 
	public double sigma;    // Standard Deviation 
	
	public double neutRating;

	//iteration number,  top words,  shuffle interval
	public int niter, ntop, shuffleLag;
	
	public int S = 3;
	public int U = 2;

	public String modelName;         //model name
	public String dir;               //data direction
	public String tData;             // text dataset filename
	public String sData;             // social dataset filename
	public String dataCharset;       // dataset charset

	/**
	 * read the option parameter
	 * @param optionFilename
	 */
	public void loadConfig(String optionFilename){
		InputStream is = null;
		try
		{
			is = new BufferedInputStream(new FileInputStream(optionFilename));
			Properties config = new Properties();
			config.load(is);
			
			niter = Integer.parseInt(config.getProperty("niter", "2000"));
			ntop = Integer.parseInt(config.getProperty("ntop"));
			shuffleLag = Integer.parseInt(config.getProperty("shuffleLag", "0"));

			//read the parameters
			alpha = Double.parseDouble(config.getProperty("alpha"));
			beta = Double.parseDouble(config.getProperty("beta", "0.1"));
			gamma = Double.parseDouble(config.getProperty("gamma", "1.5"));
			eta = Double.parseDouble(config.getProperty("eta", "0.1"));
			lamda = Double.parseDouble(config.getProperty("lamda", "0.3"));
			sigma = Double.parseDouble(config.getProperty("sigma", "0"));

			neutRating = Double.parseDouble(config.getProperty("neutRating", "3"));

			modelName = config.getProperty("modelName", "test");
			dir = config.getProperty("dir", "hdp_rating");
			tData = config.getProperty("tdata", "tdata.txt");
			sData = config.getProperty("sdata", "sdata.txt");
			dataCharset = config.getProperty("dataCharset", "GBK");

			if (beta <= 0)
				beta = 0.01;
			if (alpha <= 0)
				alpha = 1.0;

			if (niter < 0)
				niter = 1000;
			
			System.out.println("Parameters:");
			System.out.println("niter:" + niter);
			System.out.println("ntop:" + ntop);
			System.out.println("shuffleLag:" + shuffleLag);
			System.out.println("alpha:" + alpha);
			System.out.println("beta:" + beta);
			System.out.println("gamma:" + gamma);
			System.out.println("eta:" + eta);
			System.out.println("lamda:" + lamda);
			System.out.println("sigma:" + sigma);
			System.out.println("neutRating:" + neutRating);
			System.out.println("modelName:" + modelName);
			System.out.println("dir:" + dir);
			System.out.println("tData:" + tData);
			System.out.println("sData:" + sData);
			System.out.println("dataCharset:" + dataCharset);
			
		} catch(Exception e)
		{
			e.printStackTrace();
		} finally
		{
			try{is.close(); } catch(Exception e){}
		}
	}
	
	/**
	 * save the option parameter
	 * @param optionFilename
	 */
	public void saveOption(String optionFilename)
	{
		OutputStream os = null;
		
		try
		{
			os = new BufferedOutputStream(new FileOutputStream(optionFilename));
			// todo 
		} catch (Exception e)
		{
			e.printStackTrace();
		} finally
		{
			try{os.close(); } catch (Exception e){}
		}
	}

}
