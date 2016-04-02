package edu.drexel.sentiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class HDPModel {
	public final static int INITIAL_TOPIC_SIZE = 20;

	protected Random random = new Random();
	protected HDPConfig config;

	public double beta  = 0.5; // default only
	public double gamma = 1.5;
	public double alpha = 1.0;
	public double lamda = 0.5;
	public double eta = 0.5;
	public double sigma = 0;
	
	public int D;
	public int S = 3;
	public int U = 2;
	public int K = 1;
	public int X = 0;
	public int T;
	public int W;
	public int V;

	// local variables related with each document
	protected DOCState[] docStates;
	
	//counting variables not related with each document
	// # of tables by each topic
	protected int[]     m_k;               // m(k.)  m(kd)
	// # of tokens by each topic
	protected int[]     l_k;               // l(k..)
	// # of tokens by each topic and term
	protected int[][]   l_kw;
	// # of tokens by each topic, sentiment and word
	protected int[][][] l_kws;             // l(ksw)

	//authors
	// # of tokens by each topic and each author with preference u
	protected int[][][] c_kxu; 
	
	// estimated after sampling
	public double[][] theta;
	public double[][] phi;
	
	/**
	 * 
	 * @param corpus
	 */
	public void addInstances(Corpus corpus) {
		V = corpus.wDict.getSize();
		W = 0;
		D = corpus.docs.size();
		X = corpus.aDict.getSize();

		docStates = new DOCState[D];
		double r_sum = 0, r_mean = 0;
		for (int d = 0; d < D; d ++) {
			docStates[d] = new DOCState(corpus.docs.get(d), d);
			W += corpus.docs.get(d).length;
			r_sum += docStates[d].rating * docStates[d].rating;
			r_mean += docStates[d].rating;
		}

		if (config.neutRating == 0) {
			config.neutRating = (r_mean / D);
			System.out.println("NeutRating:" + config.neutRating);
		}
		if (sigma == 0) {
			sigma = r_sum / D - (r_mean / D) * (r_mean / D);
			System.out.println("sigma:" + sigma);
			if (sigma == 0)
				sigma = 1;
		}

		int k, i, j, s, u;
		DOCState docState;
		
		m_k = new int[K+1];
		l_k = new int[K+1];
		l_kw = new int[K+1][];
		l_kws = new int[K+1][][];

		c_kxu = new int[K+1][][];

		theta = new double[K][D];
		phi = new double[K][V];

		for (k = 0; k <= K; k++) { 	// var initialization done 
			l_kw[k] = new int[V];
			l_kws[k] = new int[V][S];
			c_kxu[k] = new int[X][U];
		}
		for (k = 0; k < K; k++) { 
			docState = docStates[k];

			for (i = 0; i < docState.documentLength; i++) {
				s = random.nextInt(S);
				u = random.nextInt(U);
				addWord(docState.docID, i, 0, k, s, u);
			}
		} // all topics have now one document
		
		for (j = K; j < docStates.length; j++) {
			docState = docStates[j]; 
			k = random.nextInt(K);    // all tokens have the same k in each document
			
			for (i = 0; i < docState.documentLength; i++) {
				s = random.nextInt(S); 
				u = random.nextInt(U);
				addWord(docState.docID, i, 0, k, s, u);
			}
		} // the words in the remaining documents are now assigned too
	}
	
	/**
	 * Removes a word from the bookkeeping
	 * 
	 * @param docID the id of the document the word belongs to 
	 * @param i the index of the word
	 */
	protected void removeWord(int docID, int i){
		DOCState docState = docStates[docID];
		
		int x = docState.author;
		int t = docState.words[i].t;
		int w = docState.words[i].w;
		int k = docState.tableToTopic[t];
		int s = docState.words[i].s;
		int u = docState.words[i].u;
		
		// r
		if (docState.words[i].isValidR(config.neutRating)) {
			docState.r_sum -= docState.words[i].r;
			docState.r_num --;
		}

		docState.wordCountByTable[t]--; 
		l_k[k]--; 
		l_kw[k][w]--;
		l_kws[k][w][s]--; 
		c_kxu[k][x][u]--;

		if (docState.wordCountByTable[t] == 0) { // table is removed
			T--; 
			m_k[k]--; 
			// docState.tableToTopic[t] = -1;  // ?
		}
		
		//printCounts(k, s, u, w, x);
	}
	
	/**
	 * Add a word to the bookkeeping
	 * 
	 * @param docID	docID the id of the document the word belongs to 
	 * @param i the index of the word
	 * @param table the table to which the word is assigned to
	 * @param k the topic to which the word is assigned to
	 */
	protected void addWord(int docID, int i, int table, int k, int s, int u) {
		DOCState docState = docStates[docID];
		
		int x = docState.author;
		int w = docState.words[i].w;
		
		docState.words[i].t = table; 
		docState.words[i].s = s;
		docState.words[i].u = u;
		
		// r
		docState.words[i].getR(config.neutRating);
		if (docState.words[i].isValidR(config.neutRating)) {
			docState.r_sum += docState.words[i].r;
			docState.r_num ++;
		}

		docState.wordCountByTable[table]++;

		l_k[k] ++; 
		l_kw[k][w] ++;
		l_kws[k][w][s] ++;
		c_kxu[k][x][u] ++;

		if (docState.wordCountByTable[table] == 1) { // a new table is created
			docState.numberOfTables++;
			docState.tableToTopic[table] = k;
			T ++;
			m_k[k]++; 
			
			docState.tableToTopic = Utils.ensureCapacity(docState.tableToTopic, docState.numberOfTables);
			docState.wordCountByTable = Utils.ensureCapacity(docState.wordCountByTable, docState.numberOfTables);
			if (k == K) { // a new topic is created
				K ++; 
				m_k = Utils.ensureCapacity(m_k, K); 
				l_k = Utils.ensureCapacity(l_k, K);
				l_kw = Utils.add(l_kw, new int[V], K);
				l_kws = Utils.add(l_kws, new int[V][S], K);
				c_kxu = Utils.add(c_kxu, new int[X][U], K);
			}
		}
	}
	
	/**
	 * Removes topics from the bookkeeping that have no words assigned to
	 */
	protected void defragment() {
		int[] kOldToKNew = new int[K];
		int k, newNumberOfTopics = 0;
		for (k = 0; k < K; k++) {
			if (l_k[k] > 0) {
				kOldToKNew[k] = newNumberOfTopics;
				Utils.swap(l_k, newNumberOfTopics, k);
				Utils.swap(m_k, newNumberOfTopics, k);
				Utils.swap(l_kw, newNumberOfTopics, k);
				Utils.swap(l_kws, newNumberOfTopics, k);
				Utils.swap(c_kxu, newNumberOfTopics, k);

				newNumberOfTopics++;
			} 
		}
		K = newNumberOfTopics;
		for (int j = 0; j < docStates.length; j++) 
			docStates[j].defragment(kOldToKNew);
	}
	
	/**
	 * Removes topics from the bookkeeping that have no words assigned to
	 * K >= trainModel.K
	 * But only from trainModel.K
	 */
	protected void defragment(HDPModel trainModel) {
		int[] kOldToKNew = new int[K];
		int k, newNumberOfTopics = trainModel.K;

		for (k = 0; k < trainModel.K; k++) {
			kOldToKNew[k] = k;
		}

		for (k = trainModel.K; k < K; k++) {
			if (l_k[k] > 0) {
				kOldToKNew[k] = newNumberOfTopics;
				Utils.swap(l_k, newNumberOfTopics, k);
				Utils.swap(m_k, newNumberOfTopics, k);
				Utils.swap(l_kw, newNumberOfTopics, k);
				Utils.swap(l_kws, newNumberOfTopics, k);
				Utils.swap(c_kxu, newNumberOfTopics, k);

				newNumberOfTopics++;
			} 
		}
		K = newNumberOfTopics;
		for (int j = 0; j < docStates.length; j++) 
			docStates[j].defragment(kOldToKNew);
	}

	
	/**
	 * Permute the ordering of documents and words in the bookkeeping
	 * Don't use it because it was not tested.
	 */
	protected void doShuffle(){
		List<DOCState> h = Arrays.asList(docStates);
		Collections.shuffle(h);
		docStates = h.toArray(new DOCState[h.size()]);
		for (int j = 0; j < docStates.length; j ++){
			List<WordState> h2 = Arrays.asList(docStates[j].words);
			Collections.shuffle(h2);
			docStates[j].words = h2.toArray(new WordState[h2.size()]);
		}
	}

	/**
	 * set the settings from configuration
	 * @param config
	 */
	public void setConfig(HDPConfig config) {
		this.config = config;
		
		this.gamma = config.gamma;
		this.alpha = config.alpha;
		this.beta = config.beta;
		this.eta = config.eta;
		this.lamda = config.lamda;
		this.sigma = config.sigma;

		this.U = config.U;
		this.S = config.S;
	}
	
	/**
	 * 
	 */
	public void estimateThetaPhi() {
		if (null == theta) {
			theta = new double[K][D];
		} else {
	        theta = Utils.ensureCapacity(theta, K);
		}
		if (null == phi) {
			phi = new double[K][V];
		} else {
			phi = Utils.ensureCapacity(phi, K);
		}

		int[] countsOfTopics = new int[K];
		double kalpha = K * alpha;
		
		// theta
		for (int d = 0; d < D; d ++){
			DOCState docState = docStates[d];
			for (int k = 0; k < K; k ++){
				countsOfTopics[k] = 0;
			}

			for (int t = 0; t < docState.numberOfTables; t ++) {
				countsOfTopics[docState.tableToTopic[t]] += docState.wordCountByTable[t];
			}
			
		    for (int k = 0; k < K; k ++){
			    theta[k][d] = (countsOfTopics[k] + alpha) / (docState.documentLength + kalpha);
			}
		}

		// phi
		double vbeta = V * beta;
		for (int k = 0; k < K; k ++) {
			for (int v = 0; v < V; v ++) {
				phi[k][v] = (l_kw[k][v] + beta) / (l_k[k] + vbeta);
			}
		}
	}

	/**
	 * save the model for loading in the future
	 */
	public void saveModel(HDPConfig config, Dictionary wDict, Dictionary aDict) throws Exception {
		// save the latent variables

		// D W V X S U K T 
		PrintStream file = new PrintStream(new File(config.dir, "constants.txt"), "UTF-8");
		file.format("%d", D).println();
		file.format("%d", W).println();
		file.format("%d", V).println();
		file.format("%d", X).println();
		file.format("%d", S).println();
		file.format("%d", U).println();
		file.format("%d", K).println();
		file.format("%d", T).println();
		
		file.close();
		
		// m_k
		file = new PrintStream(new File(config.dir, "m_k.txt"), "UTF-8");
		for (int k = 0; k < K; k++) {
			file.format("%d\t", m_k[k]).println();
		}
		file.close();
		
		//l_k
		file = new PrintStream(new File(config.dir, "l_k.txt"), "UTF-8");
		for (int k = 0; k < K; k++) {
			file.format("%d\t", l_k[k]).println();
		}
		file.close();
		
		//l_kw
		file = new PrintStream(new File(config.dir, "l_kw.txt"), "UTF-8");
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++)
				file.format("%d\t", l_kw[k][w]);
			file.println();
		}
		file.close();

		//l_kws
		StringBuffer sb = null;
		file = new PrintStream(new File(config.dir, "l_kws.txt"), "UTF-8");
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				sb = new StringBuffer();
				for (int s = 0; s < S; s ++) {
					sb.append(l_kws[k][w][s]);
					if (s < S - 1)
						sb.append(":");
				}
				file.format("%s ", sb.toString());
			}
			file.println();
		}
		file.close();
		
		// c_kxu
		file = new PrintStream(new File(config.dir, "c_kxu.txt"), "UTF-8");
		for (int x = 0; x < X; x++) {
		    for (int k = 0; k < K; k++) {
				sb = new StringBuffer();
				for (int u = 0; u < U; u ++) {
					sb.append(c_kxu[k][x][u]);
					if (u < U - 1)
						sb.append(":");
				}
				file.format("%s ", sb.toString());
			}
			file.println();
		}
		file.close();
		
		//为了结果结果容易阅读，还顺序排列 docs 和 words
		Arrays.sort(docStates, new Comparator<DOCState>() {
			public int compare(DOCState o1, DOCState o2) {
				return o1.docID - o2.docID;
			}
		});
		for (int d = 0; d < D; d++) {
			DOCState docState = docStates[d];
			for (int i = 0; i < docState.documentLength; i++) {
				Arrays.sort(docState.words, new Comparator<WordState>() {
					public int compare(WordState o1, WordState o2) {
						return o1.w - o2.w;
					}
				});
			}
		}
		
		// local variables  docState
		file = new PrintStream(new File(config.dir, "l_vars.txt"), "UTF-8");
		for (int d = 0; d < D; d ++) {
			DOCState docState = docStates[d];
		    for (int i = 0; i < docState.documentLength; i ++) {
				WordState wState = docState.words[i];
				file.format("%d:%d:%d:%d  ", wState.w, wState.t, wState.u, wState.s);
			}
			file.println();
		}
		file.close();

		// topics with top n words
		double vb = V * beta;
		file = new PrintStream(new File(config.dir, "topics-words.txt"), "UTF-8");
		WordTopicPair[] wtp = new WordTopicPair[V];
		for (int v = 0; v < V; v ++) {
			wtp[v] = new WordTopicPair();
		}

		for (int k = 0; k < K; k++) {
			// copy
			for (int v = 0; v < V; v ++) {
				wtp[v].v = v;
				wtp[v].count = l_kw[k][v];
			}
			Arrays.sort(wtp, new Comparator<WordTopicPair>() {
				public int compare(WordTopicPair w1, WordTopicPair w2) {
					return w2.count - w1.count;
				}
			});

			file.format("Topic:%d", k).println();
			for (int i = 0; i < Math.min(V, config.ntop); i ++)
				file.format("%s\t%d\t%5.4f ", wDict.getWord(wtp[i].v), wtp[i].count, (wtp[i].count + beta)
						/ (l_k[k] + vb)).println();
			file.println();
		}
		file.close();
		
		// author's preferences
		file = new PrintStream(new File(config.dir, "authors-preferences.txt"), "UTF-8");
		for (int x = 0; x < X; x ++) {
			file.format("Author:%s", aDict.getWord(x)).println();
			for (int k = 0; k < K; k ++) {
				file.format("%5d\t%5d\t%5d ", k, c_kxu[k][x][0], c_kxu[k][x][1]).println();
			}
			file.println();
		}

		file.close();

		// words' sentiment conditional topics
		file = new PrintStream(new File(config.dir, "words-topics-sentiments.txt"), "UTF-8");
		for (int v = 0; v < V; v ++) {
			file.format("Word: %s", wDict.getWord(v)).println();
			for (int k = 0; k < K; k ++) {
				file.format("Topic: %d", k).println();
				for (int s = 0; s < S; s ++) {
					file.format("%d", l_kws[k][v][s]).append("\t");
				}
				file.println();
			}
			file.println();
		}
		file.close();

		// predict rating and real rating
		double r_d = 0.0, mse = 0.0, l1 = 0.0;
		file = new PrintStream(new File(config.dir, "rating.txt"), "UTF-8");
		for (int d = 0; d < D; d ++) {
			DOCState docState = docStates[d];
			file.format("%5d ",docState.rating);
			
			if (0 == docState.r_num)
				r_d = config.neutRating;
			else
				r_d = docState.r_sum / docState.r_num;

			// compute f_r
			double r_u = (docState.rating - r_d);
			file.format("%4.3f", r_d);
			file.format("(%4.3f)", r_u);

			mse += r_u * r_u;
			l1 += Math.abs(r_u);

			file.println();
		}
		file.format("Mean of Square Error: %f", mse / D).println();
		file.format("Mean of Error(L1): %f", l1 / D).println();
		file.close();

		// aspect rating
		double[] r_a_sum = new double[K];
		int[]    r_a_num = new int[K];
		double[] r_a_mean = new double[K];
		file = new PrintStream(new File(config.dir, "aspect-rating.txt"), "UTF-8");
		for (int d = 0; d < D; d++) {
			DOCState docState = docStates[d];
			for (int i = 0; i < docState.words.length; i ++) {
				WordState ws = docState.words[i];
				if (ws.isValidR(config.neutRating)) {
					r_a_sum[docState.tableToTopic[ws.t]] += ws.r;
					r_a_num[docState.tableToTopic[ws.t]] ++;
				}
			}
			
			for (int k = 0; k < K; k ++) {
				if (0 == r_a_num[k])
					r_a_mean[k] = config.neutRating;
				else
					r_a_mean[k] = r_a_sum[k] / r_a_num[k];

				file.format("%1.6f\t", r_a_mean[k]);
			}
			file.println();
		}
		file.close();
	}

	/**
	 * load the model from files saved before
	 */
	public void loadModel(HDPConfig config) throws Exception {
		// load the latent variables

		// constants
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(config.dir, "constants.txt")), "UTF-8"));
		String line = null;
		// D W V X S U K T
		D = Integer.parseInt(br.readLine());
		W = Integer.parseInt(br.readLine());
		V = Integer.parseInt(br.readLine());
		X = Integer.parseInt(br.readLine());
		S = Integer.parseInt(br.readLine());
		U = Integer.parseInt(br.readLine());
		K = Integer.parseInt(br.readLine());
		T = Integer.parseInt(br.readLine());
		
		br.close();

		// m_k
		br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(config.dir, "m_k.txt")), "UTF-8"));

		m_k = new int[K + 1];
		for (int k = 0; k < K; k++) {
			m_k[k] = Integer.parseInt(br.readLine().trim());
		}
		br.close();
				
		//l_k
		br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(config.dir, "l_k.txt")), "UTF-8"));

		l_k = new int[K + 1];
		for (int k = 0; k < K; k++) {
			l_k[k] = Integer.parseInt(br.readLine().trim());
		}
		br.close();

		//l_kw
		br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(config.dir, "l_kw.txt")), "UTF-8"));

		l_kw = new int[K + 1][V];
		for (int k = 0; k < K; k++) {
			line = br.readLine();
			String[] tokens = line.split("\\s+");
			for (int v = 0; v < V; v ++)
				l_kw[k][v] = Integer.parseInt(tokens[v].trim());
		}
		br.close();

		//l_kws
		br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(config.dir, "l_kws.txt")), "UTF-8"));

		l_kws = new int[K + 1][V][S];
		for (int k = 0; k < K; k ++) {
			line = br.readLine();
			String[] tokens = line.split("\\s+");
			for (int w = 0; w < V; w ++) {
				String[] items = tokens[w].split(":");
				for (int s = 0; s < S; s ++) {
					l_kws[k][w][s] = Integer.parseInt(items[s]);
				}
			}
		}
		br.close();

		// c_kxu
		br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(config.dir, "c_kxu.txt")), "UTF-8"));
		
		c_kxu = new int[K + 1][X][U];
		for (int x = 0; x < X; x++) {
			line = br.readLine();
			String[] tokens = line.split("\\s+");
		    for (int k = 0; k < K; k++) {
		    	String[] items = tokens[k].split(":");
				for (int u = 0; u < U; u ++) {
					c_kxu[k][x][u] = Integer.parseInt(items[u]);
				}
			}
		}
		br.close();

		// local variables DOCState  vars
		br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(config.dir, "l_vars.txt")), "UTF-8"));
		docStates = new DOCState[D];
		for (int d = 0; d < D; d ++) {
			docStates[d] = new DOCState(d);
			DOCState docState = docStates[d];

			line = br.readLine();
			String[] tokens = line.split("\\s+");
			docState.documentLength = tokens.length;
			docState.words = new WordState[docState.documentLength];
		    for (int i = 0; i < docState.documentLength; i ++) {
		    	docState.words[i] = new WordState();
				WordState wState = docState.words[i];
				String[] items = tokens[i].split(":");
				wState.w = Integer.parseInt(items[0]);
				wState.t = Integer.parseInt(items[1]);
				wState.u = Integer.parseInt(items[2]);
				wState.s = Integer.parseInt(items[3]);
			}
		}
		br.close();
	}
}

class WordTopicPair {
	int v;

	int count;
}
