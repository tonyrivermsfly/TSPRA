package edu.drexel.sentiment;

public class DOCState {
	
	int docID, documentLength, numberOfTables;
	int author;
	int rating;
	int[] tableToTopic; 
    int[] wordCountByTable;
	WordState[] words;
	
	// for rating evaluation
	double r_sum;
	int    r_num;

	public DOCState(int docID) {
		this.docID = docID;
		numberOfTables = 0;
	}

	public DOCState(Document instance, int docID) {
		this.docID = docID;
	    numberOfTables = 0;
	    documentLength = instance.length;
	    author = instance.author;
	    rating = instance.rating;
	    
	    words = new WordState[documentLength];	
	    wordCountByTable = new int[2];
	    tableToTopic = new int[2];
		for (int position = 0; position < documentLength; position++) 
			words[position] = new WordState(instance.words[position], -1);
	}

	public void defragment(int[] kOldToKNew) {
	    int[] tOldToTNew = new int[numberOfTables];
	    int t, newNumberOfTables = 0;
	    for (t = 0; t < numberOfTables; t++){
	        if (wordCountByTable[t] > 0){
	            tOldToTNew[t] = newNumberOfTables;
	            tableToTopic[newNumberOfTables] = kOldToKNew[tableToTopic[t]];
	            Utils.swap(wordCountByTable, newNumberOfTables, t);
	            newNumberOfTables ++;
	        } else 
	        	tableToTopic[t] = -1;
	    }
	    numberOfTables = newNumberOfTables;
	    for (int i = 0; i < documentLength; i++)
	        words[i].t = tOldToTNew[words[i].t];
	}
}
