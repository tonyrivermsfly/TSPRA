package edu.drexel.sentiment;

public class WordState {
	int w;
	int t;   // t and z
	int u;
	int s;
	double r;

	public WordState() {
	}

	public WordState(int wordIndex, int tableAssignment){
		this.w = wordIndex;
		this.t = tableAssignment;
	}
	
	public double getR(double neutRating) {
		return r = getR(s, u, neutRating);
	}

	/**
	 * 
	 * @return
	 */
	public boolean isValidR(double neutRating) {
		return r != neutRating;
	}

	/**
	 * r
	 * @return
	 */
	public static double getR(int s, int u, double neutRating) {
		if (s == 0) {  // POS
			if (u == 1) 
				return 5;
			else 
				return (5+neutRating)/2;
		} else if (s == 1) { // NEG
			if (u == 1)
				return 1;
			else 
				return (1 + neutRating)/2;
		}
		return neutRating;
	}
}
