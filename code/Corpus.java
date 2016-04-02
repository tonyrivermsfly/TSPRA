package edu.drexel.sentiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Corpus {
	public int num_total_words;
	List<Document> docs;

	public Dictionary wDict;
	public Dictionary aDict;
	public Dictionary iDict;
		
	public Corpus(Dictionary wDict, Dictionary aDict, Dictionary iDict) {
		this.wDict = wDict;
		this.aDict = aDict;
		this.iDict = iDict;
	}

	protected void readTextData(File txtFile, String charset, boolean isTest) {
		int nd = 0, nw = 0;
		BufferedReader br = null;
		docs = new ArrayList<Document>();
		List<Integer> words = new ArrayList<Integer>();

		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(txtFile), charset));
			String line = null;

			while ((line = br.readLine()) != null) {
				String[] datas = line.split("\\s+");
				//each line is one doc  words:count
				Document doc = new Document(nd ++, datas.length);
				docs.add(doc);

				// the words
				nw = 0;
				for (int i = 0; i < datas.length; i ++) {
					String[] items = datas[i].split(":");
					items[0] = items[0].trim();
					
					if (isTest) {
						Integer wi = wDict.getID(items[0]);
						if (null == wi) {
							continue;
						} else {
							doc.words[i] = wi;
						}
					} else {
					    doc.words[i] = wDict.addWord(items[0]);
					}
					
					nw ++;

					// prior topic assigned by supervisor if possible
					if (items.length > 1) {  
						doc.topics[i] = Integer.parseInt(items[1]);
					} else {
						doc.topics[i] = -1;
					}
				}
				
				// shrink the space of doc
				doc.shrink(nw);

				num_total_words += doc.length;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		finally {
			try { br.close(); } catch (Exception e) {}
		}
	}
	protected void readSocialData(File socialFile, String charset) {
		int nd = 0;
		BufferedReader br = null;

		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(socialFile), charset));
			String line = null;
			Document doc = null;

			while ((line = br.readLine()) != null) {
				String[] datas = line.split("\\s+");
				//each line is one doc  words:count
				doc = docs.get(nd ++);

				// 1st token is review id
				
				// 2nd token is item id
				doc.item = iDict.addWord(datas[1]);
				
				// 3rd token is author id
				doc.author = aDict.addWord(datas[2]);

				// 4th token is rating
				int rating = Integer.parseInt(datas[3]);
				doc.rating = rating;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		finally {
			try { br.close(); } catch (Exception e) {}
		}
	}
	
	public void readData(File txtFile, File socialFile, String charset) {
		readTextData(txtFile, charset, false);
		readSocialData(socialFile, charset);

		System.out.println("Number of docs:        " + docs.size());
		System.out.println("Number of authors:     " + aDict.getSize());
		System.out.println("Number of terms:       " + wDict.getSize());
		System.out.println("Number of total words: " + num_total_words);
	}
	
	public void readData(File txtFile, File socialFile, String charset, boolean isTest) {
		readTextData(txtFile, charset, isTest);
		readSocialData(socialFile, charset);
		
		System.out.println("Number of docs:        " + docs.size());
		System.out.println("Number of authors:     " + aDict.getSize());
		System.out.println("Number of terms:       " + wDict.getSize());
		System.out.println("Number of total words: " + num_total_words);
	}

	public void readData(String txtFilename, String socialFilename, String charset) {
		readData(new File(txtFilename), new File(socialFilename), charset);
	}

	protected void readTextData(String filename, String charset) {
		readTextData(new File(filename), charset, false);
	}
	protected void readSocialData(String filename, String charset) {
		readSocialData(new File(filename), charset);
	}

	/**
	 * Test the loading of corpus
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("Start to load the corpus...");
		String textFilename = args[0];
		String socialFilename = args[1];
		
		Dictionary wDict = new Dictionary();
		Dictionary aDict = new Dictionary();
		Dictionary iDict = new Dictionary();
		
		Corpus c = new Corpus(wDict, aDict, iDict);
		
		c.readData(textFilename, socialFilename, "UTF-8");
		System.out.println("End of loading.");

		wDict.saveDictionary("./", "dict-v.txt");
		aDict.saveDictionary("./", "dict-a.txt");
		iDict.saveDictionary("./", "dict-i.txt");
	}
}
