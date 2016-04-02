package edu.drexel.sentiment;

public class Utils {
	public static void swap(int[] arr, int arg1, int arg2){
		   int t = arr[arg1]; 
		   arr[arg1] = arr[arg2]; 
		   arr[arg2] = t; 
	}

	public static void swap(int[][] arr, int arg1, int arg2) {
		   int[] t = arr[arg1]; 
		   arr[arg1] = arr[arg2]; 
		   arr[arg2] = t; 
	}
	
	public static void swap(int[][][] arr, int arg1, int arg2) {
		   int[][] t = arr[arg1]; 
		   arr[arg1] = arr[arg2]; 
		   arr[arg2] = t; 
	}
	
	public static double[] ensureCapacity(double[] arr, int min){
		int length = arr.length;
		if (min < length)
			return arr;
		double[] arr2 = new double[min*2];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		return arr2;
	}
	
	public static double[][] ensureCapacity(double[][] arr, int min){
		int length = arr.length;
		if (min < length)
			return arr;
		double[][] arr2 = new double[min*2][];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		for (int i = length; i < arr2.length; i ++)
			arr2[i] = new double[arr[0].length];

		return arr2;
	}
	
	public static double[][][] ensureCapacity(double[][][] arr, int min){
		int length = arr.length;
		if (min < length)
			return arr;
		double[][][] arr2 = new double[min*2][][];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		for (int i = length; i < arr2.length; i ++)
			arr2[i] = new double[arr[0].length][arr[0][0].length];
		return arr2;
	}
	
	public static double[][][][] ensureCapacity(double[][][][] arr, int min){
		int length = arr.length;
		if (min < length)
			return arr;
		double[][][][] arr2 = new double[min*2][][][];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		for (int i = length; i < arr2.length; i ++)
			arr2[i] = new double[arr[0].length][arr[0][0].length][arr[0][0][0].length];
		return arr2;
	}

	public static int[] ensureCapacity(int[] arr, int min) {
		int length = arr.length;
		if (min < length)
			return arr;
		int[] arr2 = new int[min*2];
		for (int i = 0; i < length; i++) 
			arr2[i] = arr[i];
		return arr2;
	}

	public static int[][] add(int[][] arr, int[] newElement, int index) {
		int length = arr.length;
		if (length <= index){
			int[][] arr2 = new int[index*2][];
			for (int i = 0; i < length; i++) 
				arr2[i] = arr[i];
			arr = arr2;
		}
		arr[index] = newElement;
		return arr;
	}
	
	public static int[][][] add(int[][][] arr, int[][] newElement, int index) {
		int length = arr.length;
		if (length <= index){
			int[][][] arr2 = new int[index*2][][];
			for (int i = 0; i < length; i++) 
				arr2[i] = arr[i];
			arr = arr2;
		}
		arr[index] = newElement;
		return arr;
	}
	
	public static void copy(double[][][] src, double[][][] dest) {
		for (int i = 0; i < src.length; i ++) {
			for (int j = 0; j < src[i].length; j ++) {
				for (int k = 0; k < src[i][j].length; k ++) {
					dest[i][j][k] = src[i][j][k];
				}
			}
		}
	}
}
