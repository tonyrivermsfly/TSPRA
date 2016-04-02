package edu.drexel.sentiment;

public class Document {
	public int id;                     // 
	public int item;                   //
	public int author;
	public int rating;
	public int[] words;
	public int[] topics;
	public int length;
	
	public Document(int id) {
		this.id = id;
	}

	public Document(int id, int length) {
		this.id = id;
		this.length = length;
		words = new int[length];
		topics = new int[length];
	}
	
	public void setLength(int length) {
		this.length = length;
		words = new int[length];
		topics = new int[length];
	}
	
	public void shrink(int nw) {
		if (this.length == nw)
			return;

		this.length = nw;

		int[] oWords = words;
		int[] oTopics = topics;
		words = new int[nw];
		topics = new int[nw];
		
		System.arraycopy(oWords, 0, words, 0, nw);
		System.arraycopy(oTopics, 0, topics, 0, nw);
	}
}
