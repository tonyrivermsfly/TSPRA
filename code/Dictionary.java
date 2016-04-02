package edu.drexel.sentiment;

import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import edu.ccnu.nlp.util.FileUtil;
import edu.ccnu.nlp.util.IReadLineProcessor;
import edu.ccnu.nlp.util.IWriteProcessor;

public class Dictionary {
	public Map<String,Integer> word2id;
	public Map<Integer, String> id2word;

	/**
	 * 
	 */
	public Dictionary(){
		word2id = new HashMap<String, Integer>();
		id2word = new HashMap<Integer, String>();
	}
	
	/**
	 * 
	 * @param id
	 * @return
	 */
	public String getWord(int id){
		return id2word.get(id);
	}
	
	public Integer getID(String word){
		return word2id.get(word);
	}

	/**
	 * check if this dictionary contains a specified word
	 */
	public boolean contains(String word){
		return word2id.containsKey(word);
	}

	public boolean contains(int id){
		return id2word.containsKey(id);
	}

	/**
	 * add a word into this dictionary
	 * return the corresponding id
	 */
	public int addWord(String word){
		if (!contains(word)){
			int id = word2id.size();
			
			word2id.put(word, id);
			id2word.put(id,word);

			return id;
		}
		else return getID(word);
	}

	/**
	 * read dictionary from file
	 */
	public void loadDictionary(String root, String filename) throws Exception {
		FileUtil.readTextFile(new File(root, filename), new IReadLineProcessor()
		{

			public void process(int lineNo, String line) {
				String[] tokens = line.split("\\s");
				String word = tokens[1];
				String id = tokens[0];
				int intID = Integer.parseInt(id);

				id2word.put(intID, word);
				word2id.put(word, intID);
			}
			
		}, "UTF-8");
	}
	
	public void saveDictionary(String root, String filename) throws Exception {
		FileUtil.writeTextFile(new File(root, filename), new IWriteProcessor()
		{

			public void process(PrintWriter printer) {
				for (Integer key : id2word.keySet())
				{
					printer.println(key + "\t" + id2word.get(key));
				}
			}
			
		}, "UTF-8");
	}
	
	public int getSize()
	{
		return word2id.size();
	}
}
