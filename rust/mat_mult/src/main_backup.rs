// =============================================================================
// Genetic programming for matrix multiplication (MM) algorithms
// =============================================================================
// This program evolves matrix multiplication algorithms expressed as:
//   - h-terms: products of sums of matrix entries from A and B (e.g. (a11 + a22) * (b11 - b33))
//   - c-terms: sums of h-terms that should equal the corresponding entry in C = A * B
// Fitness is measured by how closely the evolved formulas match the true product C
// on a set of random matrix triples (A, B, C). Evolution uses mutations and optional
// simulated annealing (SA).
//
// TODO:
// 1. General debugging
// 2. Accept mutations only on fitness increase
// 3. Change how mutations look for h-term names
// =============================================================================

// External crates for randomness and sparse matrices
extern crate rand;
extern crate matrix;

use std::fs::OpenOptions;
use std::fs::File;
use std::io::prelude::*;
use rand::Rng;
use matrix::prelude::*;

use std::thread;
use std::collections::HashMap;
use std::env;
use std::process::exit;
use std::time::{Duration, Instant};

// -----------------------------------------------------------------------------
// Algorithm: one candidate matrix-multiplication algorithm (genotype)
// -----------------------------------------------------------------------------
// Represents a single evolved formula: named h-terms (intermediate products) and
// c-terms (output cell formulas). Can be cloned for mutation/selection.
#[derive(Clone)]
struct Algorithm {
	/// Human-readable equation strings for each term (e.g. "h1: (a11 + a22) * (b11 - b33)")
	mult_algo: HashMap<String, String>,
	/// Per-matrix-triple: maps each term name to its integer value when evaluated on that triple
	int_maps: Vec<HashMap<String, i32>>,
	/// For each h-term name, the list of a/b subterms (e.g. ["a11", " - a22", "b33"])
	h_term_lists: HashMap<String, Vec<String>>,
	/// For each c-term name (e.g. "c12"), the list of h-terms (e.g. ["h1", " - h2"])
	c_term_lists: HashMap<String, Vec<String>>,
	/// h-terms that currently have only one a-term (used to avoid invalid "remove a" mutations)
	solo_h_list_a: Vec<String>,
	/// h-terms that currently have only one b-term (used to avoid invalid "remove b" mutations)
	solo_h_list_b: Vec<String>,
	/// c-terms that have only one or two h-terms (used to avoid invalid "remove h from c" mutations)
	solo_c_list: Vec<String>,
	fitness_cells: u32,
	fitness_difference: u32,
	/// Number of h-terms in the algorithm
	num_h: usize,
	/// If true, cap growth so new h-terms are not always added to a random c-term
	h_cap: bool,
	max_h: usize,
	/// If true, prefer keeping current algo when fitness is tied (no "neutral" acceptance)
	pos: bool,
	/// Temperature for simulated annealing (prob. of accepting worse fitness)
	temp: f64,
}

// -----------------------------------------------------------------------------
// MatMult: global configuration and data for the evolutionary run
// -----------------------------------------------------------------------------
struct MatMult {
	/// One (A, B, C) per triple; C is the true product A*B used as the target
	mat_triples: Vec<(matrix::prelude::Compressed<i32>, matrix::prelude::Compressed<i32>, matrix::prelude::Compressed<i32>)>,
	/// Matrix dimensions (rows, cols), e.g. (5, 5) for 5x5
	mat_size: (usize, usize),
	start_terms: usize,
	num_triples: usize, // Can be changed
	/// Term-size / fitness thresholds (small/medium/large)
	SMALL: u32,
	MEDIUM: u32,
	LARGE: u32,
	num_terms: usize, // Can be changed
	verbose: bool,
	cells_priority: bool, //unused
	h_added: u32,
	num_generations: usize,
	/// Use simulated annealing (accept worse solutions with probability based on temp)
	SA: bool,
	/// Number of mutations applied per generation
	num_mut: usize,
}

// -----------------------------------------------------------------------------
// init_mats: build the set of test matrix triples (A, B, C) with A*B = C
// -----------------------------------------------------------------------------
// Fills mat_trips with num_triples triples. For each triple, A and B are filled
// with random entries in [-10, 10], and C is set to the exact product A*B so we
// can measure how well an evolved algorithm matches the true result.
fn init_mats(mut mat_trips: &mut Vec<(matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>)>, num_triples: usize, mat_size: (usize, usize)){

	let mat_x: usize = mat_size.0; //Initialize matrix sizes
	let mat_y: usize = mat_size.1;
	
	let mut rng = rand::thread_rng();
	let z: i32 = rng.gen_range(-10..11); //Generate random ints between -10 and 10
	
	let mut m1: matrix::prelude::Compressed<i32> = Compressed::zero((5,5)); //Initialize 3 matrices
	let mut m2: matrix::prelude::Compressed<i32> = Compressed::zero((5,5));
	let mut m3: matrix::prelude::Compressed<i32> = Compressed::zero((5,5));
	
	for triple in 0..num_triples{
		// Fill A and B with random entries in [-10, 10]; compute C = A*B
		for x in 0..mat_x{
			for y in 0..mat_y{
			
				m1.set((x,y), rng.gen_range(-10..11));
				m2.set((x,y), rng.gen_range(-10..11));
			}	
		}
	
		// Compute C = A * B (matrix multiplication)
		for x in 0..mat_x{
			for y in 0..mat_x{
				let mut current = 0;
				for z in 0..mat_y{
					current += m1.get((x,z)) * m2.get((z,y));
				}
				m3.set((x,y), current);
			}
		}
		mat_trips.push((m1.clone(),m2.clone(),m3.clone()));
	}
}	

//////

// -----------------------------------------------------------------------------
// eval_single_c: evaluate one c-term (output cell formula) to an integer
// -----------------------------------------------------------------------------
// c_list holds the h-term names in this c-term (e.g. ["h1", " - h2"]). Each is
// looked up in int_map (already evaluated). Signs are respected; result is
// written into int_map under c_key (e.g. "c12").
fn eval_single_c(int_map: &mut HashMap<String, i32>, c_list: Vec<String>, c_key: &String){
	let mut res = 0;
	let mut h_int_vec = [].to_vec();
				
	for term in c_list{ //Iterate through each h-term in the c-term
				
		let mut read_term = term.clone();
		let mut negative = false;
					
		if read_term.contains("-") { negative = true }
					
		read_term = read_term.replace(" - ", "");
					
		if negative{
			h_int_vec.push(-int_map[&read_term]); // modified
		} else {
			h_int_vec.push(int_map[&read_term])
		}	
	}
	res = h_int_vec.iter().sum();
	int_map.insert(c_key.to_string(), res);	
		
}

// -----------------------------------------------------------------------------
// eval_single_h: evaluate one h-term to an integer using matrix entries from one triple
// -----------------------------------------------------------------------------
// An h-term is (sum of a-terms) * (sum of b-terms). h_list contains subterms like
// "a11", " - a22", "b33". Values are read from mat_triple.0 (A) and mat_triple.1 (B);
// the product of the two sums is stored in int_map under h_key.
fn eval_single_h(int_map: &mut HashMap<String, i32>, h_list: Vec<String>, h_key: String, mat_triple: (matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>)){

	let mut res = 0;
	let now = Instant::now();
	let mut a_res = 0;
	let mut b_res = 0;
						
	let mut a_vec = [].to_vec();
	let mut a_int_vec: Vec<i32> = [].to_vec();
	let mut b_vec = [].to_vec();
	let mut b_int_vec:  Vec<i32>  = [].to_vec();
					
	for term in h_list{ //Iterate through all terms in the equation and replace with corresponding int
					
		let mut negative = false;
		let mut read_term = term.clone();
						
		if term.contains("-"){ negative = true; }
					
		read_term = read_term.replace(" - ", "");
		let x_coord = read_term[1..2].to_string().parse::<usize>().unwrap()-1;
		let y_coord = read_term[2..3].to_string().parse::<usize>().unwrap()-1;
						
		if term.contains("a"){ 
			a_vec.push(read_term);
			if negative{
				a_int_vec.push( (mat_triple.0.get((x_coord, y_coord))) - (2*(mat_triple.0.get((x_coord, y_coord)))))
			} else {
				a_int_vec.push( mat_triple.0.get((x_coord, y_coord)))
			}
		} else if term.contains("b"){ 
			b_vec.push(read_term);

			if negative{
				b_int_vec.push( (mat_triple.1.get((x_coord, y_coord))) - (2*(mat_triple.1.get((x_coord, y_coord)))))
			} else {
				b_int_vec.push( mat_triple.1.get((x_coord, y_coord)))
			}
		}
	}
	a_res = a_int_vec.iter().sum();
	b_res = b_int_vec.iter().sum();
	int_map.insert(h_key.clone(), a_res*b_res);
}

//////

// -----------------------------------------------------------------------------
// init_maps: evaluate all h-terms and c-terms on every matrix triple
// -----------------------------------------------------------------------------
// Builds one int_map per triple in algo.int_maps. For each triple, first all
// h-terms are evaluated (they depend only on A and B), then all c-terms (they
// depend on h-term values). Used when initializing a new algorithm or after
// loading one from disk.
fn init_maps(mut algo: &mut Algorithm, matmult: &MatMult, mat_triples: &Vec<(matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>)>, num_terms: usize, term_size: u32, mat_size: (usize, usize)){

	let mat_triples = &matmult.mat_triples;
	
	for mat_triple in 0..mat_triples.len() {
	
		let mut int_map: HashMap<String, i32> = HashMap::new();
		algo.int_maps.push(int_map);

		for h in algo.h_term_lists.keys(){ //Iterate through all h-terms
				
			eval_single_h(&mut algo.int_maps[mat_triple], algo.h_term_lists[h].clone(), h.to_string(), mat_triples[mat_triple].clone());
		}
		
		for c in algo.c_term_lists.keys(){
			
				eval_single_c(&mut algo.int_maps[mat_triple], algo.c_term_lists[c].clone(), &c.to_string());
		}
		
	}
	
}

// -----------------------------------------------------------------------------
// update_maps: re-evaluate only the terms that were changed by a mutation
// -----------------------------------------------------------------------------
// keys lists term names (e.g. "h3", "c12") whose formulas were mutated. For each
// triple we re-evaluate those terms and any c-terms that depend on a changed
// h-term, so int_maps stay in sync without recomputing everything.
fn update_maps(algo: &mut Algorithm, mat_triples: &Vec<(matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>)>, num_terms: usize, term_size: u32, mat_size: (usize, usize), keys: Vec<String>){

	for mat_triple in 0..mat_triples.len(){
	
		for key in &keys{
		
			if key.contains("h"){
			
				//println!("Re-evaluating {} to {:?}", key, algo.h_term_lists[key]);
				eval_single_h(&mut algo.int_maps[mat_triple], algo.h_term_lists[key].clone(), key.to_string(), mat_triples[mat_triple].clone());
				
				for row in 1..mat_size.0+1{
					for col in 1..mat_size.1+1{
						let mut current_c = String::from("c");
						current_c.push_str(row.to_string().as_str());
						current_c.push_str(col.to_string().as_str());
						
						let mut negative = String::from(" - ");
						negative.push_str(&current_c);
						
						for term in &algo.c_term_lists[&current_c]{
							
							if term == &current_c || term == &negative{
								//println!("Re-evaluating {} to {:?}", &current_c, algo.c_term_lists[&current_c]);
								eval_single_c(&mut algo.int_maps[mat_triple], algo.c_term_lists[&current_c].clone(), &current_c);
							}							
						}
					}
				}
					
			
			} else if key.contains("c"){
			
				//println!("Re-evaluating {} to {:?}", key, algo.c_term_lists[key]);
				eval_single_c(&mut algo.int_maps[mat_triple], algo.c_term_lists[key].clone(), &key.to_string());
			
			}
		
		}
	
	}
	
}

// -----------------------------------------------------------------------------
// get_fitness: compare algorithm output to true C on all triples; lower is better
// -----------------------------------------------------------------------------
// Uses existing int_maps (must be up to date). Returns (avg_cell_difference,
// avg_wrong_cell_count): (0.0, 0.0) means the algorithm is correct on all triples.
fn get_fitness(mut algo: &mut Algorithm, matmult: &MatMult, mat_triples: &Vec<(matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>)>, num_terms: usize, term_size: u32, mat_size: (usize, usize)) -> (f64, f64){

	let mut fitness_difference: f64 = 0.0;
	let mut fitness_difference_final: f64 = 0.0;
	let mut fitness_cells: f64 = 0.0;
	let mut mat_triples = &matmult.mat_triples;
	
	for x in 0..mat_triples.len(){	//Iterate though each matrix triple
		
		fitness_difference = 0.0;
		
		for row in 1..mat_size.0+1{
			for col in 1..mat_size.1+1{	//Iterate through each c-term
			
				let mut current_c = String::from("c");
				current_c.push_str(row.to_string().as_str());
				current_c.push_str(col.to_string().as_str());
				
				let res = algo.int_maps[x][&current_c];
				
				if res != mat_triples[x].2.get((row-1, col-1)){

					fitness_difference += (mat_triples[x].2.get((row-1,col-1)) - res).abs() as f64;
					fitness_cells += 1.0
				}
			}
		}
		fitness_difference /= 25.0;
		fitness_difference_final += fitness_difference;
	}
	fitness_cells /= mat_triples.len() as f64;
	fitness_difference_final = fitness_difference_final / mat_triples.len() as f64;
	
	(fitness_difference_final.into(), fitness_cells as f64)

}

// -----------------------------------------------------------------------------
// print_mats: pretty-print one matrix triple (A, then B, then C) to stdout
// -----------------------------------------------------------------------------
fn print_mats(mat_trip: &(matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>,matrix::prelude::Compressed<i32>)){

	for x in 0..5{
		println!("{} {} {} {} {}", mat_trip.0.get((x,0)), mat_trip.0.get((x,1)), mat_trip.0.get((x,2)), mat_trip.0.get((x,3)), mat_trip.0.get((x,4)))
	}
	println!("\n");
	for x in 0..5{
		println!("{} {} {} {} {}", mat_trip.1.get((x,0)), mat_trip.1.get((x,1)), mat_trip.1.get((x,2)), mat_trip.1.get((x,3)), mat_trip.1.get((x,4)))
	}
	println!("\n");
	for x in 0..5{
		println!("{} {} {} {} {}", mat_trip.2.get((x,0)), mat_trip.2.get((x,1)), mat_trip.2.get((x,2)), mat_trip.2.get((x,3)), mat_trip.2.get((x,4)))
	}
	println!("\n");
}

// -----------------------------------------------------------------------------
// print_mat3_algo: print the algorithm’s evaluated c-terms as a 5x5 matrix
// -----------------------------------------------------------------------------
// Uses the first int_map (first triple) to show c11..c55 in matrix layout.
fn print_mat3_algo(algo: &Algorithm){

	for row in 1..6{
		let mut c1 = String::from("c");
		c1.push_str(row.to_string().as_str());
		c1.push_str(1.to_string().as_str());
		let mut c2 = String::from("c");
		c2.push_str(row.to_string().as_str());
		c2.push_str(2.to_string().as_str());
		let mut c3 = String::from("c");
		c3.push_str(row.to_string().as_str());
		c3.push_str(3.to_string().as_str());
		let mut c4 = String::from("c");
		c4.push_str(row.to_string().as_str());
		c4.push_str(4.to_string().as_str());
		let mut c5 = String::from("c");
		c5.push_str(row.to_string().as_str());
		c5.push_str(5.to_string().as_str());
		println!("{} {} {} {} {}", algo.int_maps[0][&c1],algo.int_maps[0][&c2],algo.int_maps[0][&c3],algo.int_maps[0][&c4],algo.int_maps[0][&c5]);
	}
}

// -----------------------------------------------------------------------------
// rand_algo: create a random matrix-multiplication algorithm from scratch
// -----------------------------------------------------------------------------
// term_size = number of a/b subterms per h-term and number of h-terms per c-term.
// num_terms = number of h-terms. Creates h1..h_num_terms and c11..c_ij for the
// given mat_size, all with random structure (signs and indices).
fn rand_algo(term_size: u32, num_terms: usize, mat_size: (usize, usize)) -> Algorithm{

	let mut algo = Algorithm{
		mult_algo: HashMap::new(),
		h_term_lists: HashMap::new(),
		c_term_lists: HashMap::new(),
		int_maps: [].to_vec(),
		solo_h_list_a: [].to_vec(),
		solo_h_list_b: [].to_vec(),
		solo_c_list: [].to_vec(),
		fitness_cells: 1000,
		fitness_difference: 1000,
		num_h: 0,
		h_cap: false,
		max_h: 2*num_terms,
		pos: false,
		temp: 1.0,
	};
	
	for x in 1..num_terms+1{
	
		let mut h_term_list = make_h_list(term_size);
		let mut h_term = make_h(&h_term_list);
		let mut h_term_name = String::from("h");
		h_term_name.push_str(x.to_string().as_str());
		
		algo.h_term_lists.insert(h_term_name.clone(), h_term_list);
		algo.mult_algo.insert(h_term_name, h_term);
	}
	
	for x in 1..mat_size.0+1{
		for y in 1..mat_size.1+1{
			
			let mut c_term_list = make_c_list(term_size, num_terms);
			
			let mut c_term = make_c(&c_term_list);
			let mut c_term_name = String::from("c");
			
			
			c_term_name.push_str(x.to_string().as_str());
			c_term_name.push_str(y.to_string().as_str());
			
			algo.c_term_lists.insert(c_term_name.clone(), c_term_list);
			algo.mult_algo.insert(c_term_name, c_term);
			
		}
	}
	
	algo
}

// -----------------------------------------------------------------------------
// print_algo: print the algorithm’s equations to stdout and append them to write_file
// -----------------------------------------------------------------------------
// Format: "START ALGORITHM", then each h-term and c-term as "name: equation".
fn print_algo(algo: &Algorithm, num_terms: usize, mat_size: (usize, usize), mut write_file: File){

	write_file.write("\n".as_bytes());
	write_file.write("START ALGORITHM".as_bytes());
	write_file.write("\n".as_bytes());

	for x in 1..num_terms+1{
	
		let mut term_name = String::from("h");
		term_name.push_str(x.to_string().as_str());
		if algo.mult_algo.contains_key(&term_name){
			println!("{}: {}", term_name, algo.mult_algo[&term_name]);
			write_file.write(term_name.to_string().as_bytes());
			write_file.write(": ".as_bytes());
			write_file.write(algo.mult_algo[&term_name].as_bytes());
			write_file.write("\n".as_bytes());
		}
	}
	
	println!("\n");
	
	for x in 1..mat_size.0+1{
		for y in 1..mat_size.1+1{
		
				let mut term_name = String::from("c");
				term_name.push_str(x.to_string().as_str());
				term_name.push_str(y.to_string().as_str());
				
				println!("{}: {}", term_name, algo.mult_algo[&term_name]);
				write_file.write(term_name.to_string().as_bytes());
				write_file.write(": ".as_bytes());
				write_file.write(algo.mult_algo[&term_name].as_bytes());
				write_file.write("\n".as_bytes());
		
		}
	}

}

// -----------------------------------------------------------------------------
// print_int_maps: print evaluated values for all h-terms and c-terms (first triple only)
// -----------------------------------------------------------------------------
fn print_int_maps(algo: &Algorithm, num_terms: usize, mat_size: (usize, usize)){

	for x in 1..num_terms+1{
	
		let mut term_name = String::from("h");
		term_name.push_str(x.to_string().as_str());
		if algo.mult_algo.contains_key(&term_name){
			println!("{}: {}", term_name, algo.int_maps[0][&term_name]);
		}
	}
	
	println!("\n");
	
	for x in 1..mat_size.0+1{
		for y in 1..mat_size.1+1{
		
				let mut term_name = String::from("c");
				term_name.push_str(x.to_string().as_str());
				term_name.push_str(y.to_string().as_str());
				
				println!("{}: {}", term_name, algo.int_maps[0][&term_name]);
		
		}
	}

}

// -----------------------------------------------------------------------------
// make_ab_term: create one random a-term or b-term (e.g. "a12", " - b34")
// -----------------------------------------------------------------------------
// Randomly chooses a or b, optional leading " - ", and indices in 1..6 (for 5x5).
fn make_ab_term(term_size: u32) -> String{
	
	let mut rng = rand::thread_rng();
	let z: i32 = rng.gen_range(0..2);
	
	let mut term = String::from("");
	
	if z == 0{
		term.push_str(" - ");	
	}
	
	let z: i32 = rng.gen_range(0..2);
	
	if z == 0{ term.push('a') } else { term.push('b') }
	
	let z: i32 = rng.gen_range(1..6);
	
	term.push_str(z.to_string().as_str());
	
	let z: i32 = rng.gen_range(1..6);
	
	term.push_str(z.to_string().as_str());
	
	term
	
}

// -----------------------------------------------------------------------------
// make_h_list: build the list of a/b subterms for one h-term
// -----------------------------------------------------------------------------
// Returns a vector of term_size strings from make_ab_term. If the result would
// be one-sided (only a's or only b's), retries until both appear (valid h-term).
fn make_h_list(term_size: u32) -> Vec<String>{

	let mut h_term_list: Vec<String> = [].to_vec();
	
	for x in 0..term_size{
		h_term_list.push(make_ab_term(term_size));
	}

	
	if one_sided_h(&h_term_list){
		h_term_list = make_h_list(term_size);
	}

	h_term_list

}

// -----------------------------------------------------------------------------
// make_h: turn an h-term list into a single equation string "(sum_a) * (sum_b)"
// -----------------------------------------------------------------------------
// Groups a-terms and b-terms, formats with + / - and parentheses, e.g.
// "(a11 + a22 - a33) * (b11 + b44)".
fn make_h(h_term_list: &Vec<String>) -> String{

	let mut h_term_a = String::from("");
	let mut h_term_b = String::from("");

	for x in 0..h_term_list.len(){
		if h_term_list[x].contains("a") && !h_term_list[x].contains("-"){
			h_term_a.push_str(" + ");
			h_term_a.push_str(h_term_list[x].as_str())
		}
	}
	
	for x in 0..h_term_list.len(){
		if h_term_list[x].contains("a") && h_term_list[x].contains("-"){
			h_term_a.push_str(h_term_list[x].as_str())
		}
	}
	
	for x in 0..h_term_list.len(){
		if h_term_list[x].contains("b") && !h_term_list[x].contains("-"){
			h_term_b.push_str(" + ");
			h_term_b.push_str(h_term_list[x].as_str())
		}
	}
	for x in 0..h_term_list.len(){
		if h_term_list[x].contains("b") && h_term_list[x].contains("-"){
			h_term_b.push_str(h_term_list[x].as_str());
		}
	}
	
	h_term_a = h_term_a.trim().to_string();
	h_term_b = h_term_b.trim().to_string();
	if h_term_a.as_bytes()[0] as char == '+' {h_term_a = h_term_a[1..h_term_a.len()].to_string()}
	if h_term_b.as_bytes()[0] as char == '+' {h_term_b = h_term_b[1..h_term_b.len()].to_string()}
	h_term_a = h_term_a.trim().to_string();
	h_term_b = h_term_b.trim().to_string();
	let mut temp = String::from('('); temp.push_str(h_term_a.as_str()); temp.push(')'); h_term_a = temp;
	let mut temp = String::from('('); temp.push_str(h_term_b.as_str()); temp.push(')'); h_term_b = temp;
	if h_term_a.as_bytes()[2] as char == ' ' {h_term_a = h_term_a.replacen(" ", "", 1)}
	if h_term_b.as_bytes()[2] as char == ' ' {h_term_b = h_term_b.replacen(" ", "", 1)}
	
	let mut h_term = String::from(h_term_a);
	h_term.push_str(" * ");
	h_term.push_str(h_term_b.as_str()); 
	
	h_term
	
}

// -----------------------------------------------------------------------------
// make_c_list: build the list of h-terms (with optional minus) for one c-term
// -----------------------------------------------------------------------------
// Returns term_size entries, each "hK" or " - hK" for random K in 1..num_terms+1.
fn make_c_list(term_size: u32, num_terms: usize) -> Vec<String>{

	let mut c_term_list: Vec<String> = [].to_vec();
	
	let mut rng = rand::thread_rng();

	for x in 0..term_size{
	
		let z: i32 = rng.gen_range(0..2);
		let mut term = String::from("");
		
		if z == 0{ term.push_str(" - ")}
		term.push('h');
		
		let z: usize = rng.gen_range(1..num_terms+1);
		term.push_str(z.to_string().as_str());
		
		c_term_list.push(term);
	}
		
	c_term_list

}

// -----------------------------------------------------------------------------
// make_c: turn a c-term list into a single equation string (sum of h-terms)
// -----------------------------------------------------------------------------
// Positive h-terms get " + " prefix, negative ones are written as " - hK".
fn make_c(c_term_list: &Vec<String>) -> String{

	let mut c_term = String::from("");
	
	for x in 0..c_term_list.len(){
		if !c_term_list[x].contains("-"){
			c_term.push_str(" + ");
			c_term.push_str(c_term_list[x].as_str())
		}
	}
	
	for x in 0..c_term_list.len(){
		if c_term_list[x].contains("-"){
			c_term.push_str(c_term_list[x].as_str())
		}
	}
	
	c_term = c_term.trim().to_string();
	//println!("Here is the error: {:?}", c_term_list);
	if c_term.as_bytes()[0] as char == '+'{c_term = c_term[1..c_term.len()].to_string()}
	c_term = c_term.trim().to_string();
	c_term.push(' ');
	if c_term.as_bytes()[0] as char == ' '{c_term = c_term.replacen(" ", "", 1)}
	
	c_term

}

// -----------------------------------------------------------------------------
// one_sided_h: true if the h-term has only a-terms or only b-terms (invalid)
// -----------------------------------------------------------------------------
// A valid h-term must have at least one a and one b so it represents a product
// of two sums. Used by make_h_list to reject and retry one-sided terms.
fn one_sided_h(h_term_list: &Vec<String>) -> bool{

	let mut num_a = 0;
	let mut num_b = 0;
	
	for x in 0..h_term_list.len(){

		if h_term_list[x].contains("a"){ num_a += 1} else if h_term_list[x].contains("b"){num_b +=1}
	}
	
	if num_a == 0 || num_b == 0 {return true;} 
	
	false

}


// -----------------------------------------------------------------------------
// mutate: apply one random mutation to the algorithm; return updated algo and metadata
// -----------------------------------------------------------------------------
// Picks mutation type 1..8 uniformly, applies it, and returns:
//   (algo, num_terms, mutation_type, no_mutate, keys)
// keys = list of term names whose expressions changed (for update_maps).
// no_mutate = true if the mutation could not be applied (e.g. nothing left to remove).
//
// Mutation types:
//   1: Add new h-term and optionally add it to a random c-term
//   2: Remove an existing h-term from the algorithm and from all c-terms that use it
//   3: Add a random (+/-) a-term to a random h-term
//   4: Add a random (+/-) b-term to a random h-term
//   5: Remove one a-term from an h-term that has more than one a-term
//   6: Remove one b-term from an h-term that has more than one b-term
//   7: Add a random (+/-) h-term to a random c-term
//   8: Remove one h-term from a c-term that has more than two h-terms
fn mutate(mut algo: &mut Algorithm, term_size: u32, mut num_terms: usize, mat_size: (usize, usize), num_triples: usize, verbose: bool) -> (&Algorithm, usize, usize, bool, Vec<String>){
	
	let mut rng = rand::thread_rng();
	//let mut mutation_type: usize = 8;
	let mut mutation_type: usize = rng.gen_range(1..9);
	//println!("MUT {}, keys {}", mutation_type, algo.h_term_lists.keys().len());
	let mut no_mutate: bool = false;
	let mut keys = [].to_vec();
	
	if mutation_type == 1{ // Add new h-term; optionally add it to one random c-term
	
		if algo.num_h == algo.max_h {
		
			return (algo, num_terms, mutation_type, no_mutate, keys)
		
		}

		let mut h_to_add = String::from("h");			//Initialize the name of the new h-term
		h_to_add.push_str((num_terms+1).to_string().as_str());
		
		keys.push(h_to_add.clone());				//Add the h-term to the list of terms to be evaluated and updated in the int_maps
		
		let mut new_h_list = make_h_list(term_size);		//Initialize the h-term's list and String expression
		let mut new_h = make_h(&new_h_list);			
		
		algo.h_term_lists.insert(h_to_add.clone(), new_h_list);	//Add the h-term's list and String expression to the current MM algorithm
		algo.mult_algo.insert(h_to_add.clone(), new_h);
		
		//Also add this new h to one of the c's
		
		algo.num_h +=1;
		
		if !algo.h_cap{
		
			let mut row: usize = rng.gen_range(1..6);		//Select a random c-term
			let mut col: usize = rng.gen_range(1..6);
			
			let mut c_to_add_to = String::from("c");		
			c_to_add_to.push_str(row.to_string().as_str());
			c_to_add_to.push_str(col.to_string().as_str());
			
			let mut random: usize = rng.gen_range(0..2);
			let mut new_h_name = String::from(" - "); 
			
			keys.push(c_to_add_to.clone());				//Add the c-term to the list of terms to be evaluated and updated in the int_maps
			
			if random == 0{ new_h_name.push_str(h_to_add.as_str()); h_to_add = new_h_name.clone();}	//Decide whether the term will be negative in the c-term's equation
			
			if verbose{
				println!("Adding {}", h_to_add);
			}
			
			algo.c_term_lists.get_mut(&c_to_add_to).map(|val| val.push(h_to_add));	//Add the new h-term to the selected c-term's list
			algo.mult_algo.insert(c_to_add_to.clone(), make_c(&algo.c_term_lists[&c_to_add_to])); //Update the selected c-term's equation
		
		}
		
		num_terms += 1;
		
	} else if mutation_type == 2{ // Remove one h-term from the algorithm and from every c-term that references it
	
		let mut unremovable_list: Vec<String> = [].to_vec();	/*A list which will contain the names of all h-terms which cannot be removed,
									  because it is the only unique h-term in a c-term's equation.*/
		
		for row in 1..(mat_size.0)+1{				//Iterate through all c-terms, identifying 
			for col in 1..(mat_size.1)+1{
				let mut current_c = String::from("c");
				current_c.push_str(row.to_string().as_str());
				current_c.push_str(col.to_string().as_str());
				
				let mut count = 0;
				let mut h = &algo.c_term_lists[&current_c][0].replace(" - ", "");
				let mut negative_h = String::from(" - ");
				negative_h.push_str(h.as_str());
					
				for x in 0..algo.c_term_lists[&current_c].len(){
					
					if &algo.c_term_lists[&current_c][x] == h || algo.c_term_lists[&current_c][x] == negative_h{count +=1}
				}
					
				if count == algo.c_term_lists[&current_c].len() && !unremovable_list.contains(h){ 
					
					unremovable_list.push(h.to_string()); 
				}
			}
		}
		
		let mut term_index = rng.gen_range(0..algo.h_term_lists.keys().len());
		let h_keys: Vec<&str> = algo.h_term_lists.keys().map(|k| k.as_str()).collect();
		let mut h_to_remove = h_keys[term_index].to_string();
		
		while !algo.mult_algo.contains_key(&h_to_remove) || unremovable_list.contains(&h_to_remove){ //If the 
			
			if unremovable_list.len() == h_keys.len(){
				no_mutate = true;
				return (algo, num_terms, mutation_type,  no_mutate, keys)
			}	
			//if unremovable_list.contains(&h_to_remove){ println!("UNREMOVABLE: {} , key {:?} , unremovable {:?}", h_to_remove, h_keys, unremovable_list); }
			//println!("unremovable length: {}, keys length: {}", unremovable_list.len(), h_keys.len());
			//println!("Unremovable: {}",!algo.mult_algo.contains_key(&h_to_remove), unremovable_list.contains(&h_to_remove));

			term_index = rng.gen_range(0..algo.h_term_lists.keys().len());
			h_to_remove = h_keys[term_index].to_string();
		}		
		
		if verbose{
			println!("Removing {}", h_to_remove);
		}
		
		for x in 0..num_triples{
			algo.int_maps[x].remove(&h_to_remove);
		}
		
		algo.mult_algo.remove(&h_to_remove);
		algo.h_term_lists.remove(&h_to_remove);
		
		algo.num_h -=1;
		
		if algo.solo_h_list_a.contains(&h_to_remove){
			let mut index = algo.solo_h_list_a.iter().position(|x| *x == h_to_remove).unwrap();
			algo.solo_h_list_a.remove(index);
		}
		if algo.solo_h_list_b.contains(&h_to_remove){
			let mut index = algo.solo_h_list_b.iter().position(|x| *x == h_to_remove).unwrap();
			algo.solo_h_list_b.remove(index);
		}
		
		for row in 1..(mat_size.0)+1{
			for col in 1..(mat_size.1)+1{
				let mut current_c = String::from("c");
				current_c.push_str(row.to_string().as_str());
				current_c.push_str(col.to_string().as_str());
				let mut neg_check = String::from(" - ");
				neg_check.push_str(h_to_remove.as_str());
				
				if algo.c_term_lists[&current_c].contains(&h_to_remove){
					//println!("Removing {} from {}: {:?}", &h_to_remove, &current_c, algo.c_term_lists[&current_c]);
					//println!("{:?}", algo.c_term_lists[&current_c]);
					keys.push(current_c.clone());
					
					//println!("C list in question: {:?}", algo.c_term_lists[&current_c]);
						
					while algo.c_term_lists[&current_c].contains(&h_to_remove){
						let mut index = algo.c_term_lists[&current_c].iter().position(|x| *x == h_to_remove).unwrap();
						algo.c_term_lists.get_mut(&current_c).map(|val| val.remove(index));
					}
					
					//println!("C list in question 2: {:?}", algo.c_term_lists[&current_c]);
					
					//println!("Unremovable list: {:?}", unremovable_list);

					algo.mult_algo.insert(current_c.clone(), make_c(&algo.c_term_lists[&current_c]));
				}
				
				if algo.c_term_lists[&current_c].contains(&neg_check){
					//println!("Removing from {}", &current_c);
					keys.push(current_c.clone());
					
					while algo.c_term_lists[&current_c].contains(&neg_check){
						let mut index = algo.c_term_lists[&current_c].iter().position(|x| *x == neg_check).unwrap();
						algo.c_term_lists.get_mut(&current_c).map(|val| val.remove(index));

					}
					algo.mult_algo.insert(current_c.clone(), make_c(&algo.c_term_lists[&current_c]));
				}
			}
		}
	} else if mutation_type == 3{ // Add a random (+/-) a-term to a random h-term
		let mut random = rng.gen_range(0..2);
		let mut a_to_add = String::from("");
		
		if random == 0{ a_to_add.push_str(" - ")}
		a_to_add.push('a');
		
		let mut row = rng.gen_range(1..6);
		let mut col = rng.gen_range(1..6);
		
		a_to_add.push_str(row.to_string().as_str());
		a_to_add.push_str(col.to_string().as_str());
		
		let mut h_no = rng.gen_range(1..num_terms+1);
	
		let mut h_to_add_to = String::from("h");
		h_to_add_to.push_str(h_no.to_string().as_str());
		
		while !algo.mult_algo.contains_key(&h_to_add_to){
			let mut h_no = rng.gen_range(1..num_terms+1);
	
			h_to_add_to = String::from("h");
			h_to_add_to.push_str(h_no.to_string().as_str());
		}
		
		if verbose{
			println!("ADDING {} TO: {}", a_to_add, h_to_add_to);
		}
		
		keys.push(h_to_add_to.clone());
		
		algo.h_term_lists.get_mut(&h_to_add_to).map(|val| val.push(a_to_add));
		
		algo.mult_algo.insert(h_to_add_to.clone(), make_h(&algo.h_term_lists[&h_to_add_to]));
		
		if algo.solo_h_list_a.contains(&h_to_add_to){
			let mut index = algo.solo_h_list_a.iter().position(|x| *x == h_to_add_to).unwrap();
			algo.solo_h_list_a.remove(index);
		}
	
	} else if mutation_type == 4{ // Add a random (+/-) b-term to a random h-term
		let mut random = rng.gen_range(0..2);
		let mut b_to_add = String::from("");
		
		if random == 0{ b_to_add.push_str(" - ")}
		b_to_add.push('b');
		
		let mut row = rng.gen_range(1..6);
		let mut col = rng.gen_range(1..6);
		
		b_to_add.push_str(row.to_string().as_str());
		b_to_add.push_str(col.to_string().as_str());
		
		let mut h_no = rng.gen_range(1..num_terms+1);
	
		let mut h_to_add_to = String::from("h");
		h_to_add_to.push_str(h_no.to_string().as_str());
		
		while !algo.mult_algo.contains_key(&h_to_add_to){
			let mut h_no = rng.gen_range(1..num_terms+1);
	
			h_to_add_to = String::from("h");
			h_to_add_to.push_str(h_no.to_string().as_str());
		}
		
		if verbose{
			println!("ADDING {} TO: {}", b_to_add, h_to_add_to);
		}
		
		keys.push(h_to_add_to.clone());
		
		algo.h_term_lists.get_mut(&h_to_add_to).map(|val| val.push(b_to_add));
		
		algo.mult_algo.insert(h_to_add_to.clone(), make_h(&algo.h_term_lists[&h_to_add_to]));
		
		if algo.solo_h_list_b.contains(&h_to_add_to){
			let mut index = algo.solo_h_list_b.iter().position(|x| *x == h_to_add_to).unwrap();
			algo.solo_h_list_b.remove(index);
		}
	
	} else if mutation_type == 5{ // Remove one a-term from an h-term that has at least two a-terms
		let mut h_to_remove_from = String::from("");
		let mut valid = false;
		
		while !valid{
		
			h_to_remove_from.push('h');
			let mut h_no = rng.gen_range(1..num_terms+1);
			h_to_remove_from.push_str(h_no.to_string().as_str());
			
			while !algo.mult_algo.contains_key(&h_to_remove_from){
				h_to_remove_from = String::from("h");
				let mut h_no = rng.gen_range(1..num_terms+1);
				h_to_remove_from.push_str(h_no.to_string().as_str());
			}
			
			let mut num_a = 0;
		
			for term in 0..algo.h_term_lists[&h_to_remove_from].len(){
			
				if algo.h_term_lists[&h_to_remove_from][term].contains("a"){
					num_a +=1;
				}
				
				if num_a > 1{
					valid = true;
				} else if !algo.solo_h_list_a.contains(&h_to_remove_from){
					algo.solo_h_list_a.push(h_to_remove_from.clone());
				}
				
				if algo.solo_h_list_a.len() == algo.h_term_lists.keys().len(){
					no_mutate = true;
					return (algo, num_terms, mutation_type, no_mutate, keys)
				}
			}
		}
		
		keys.push(h_to_remove_from.clone());
		
		let mut a_terms = [].to_vec();
		
		for x in 0..algo.h_term_lists[&h_to_remove_from].len(){
		
			if algo.h_term_lists[&h_to_remove_from][x].contains("a"){
				a_terms.push(&algo.h_term_lists[&h_to_remove_from][x]);
			}
		}
		
		let mut picked_term_i = rng.gen_range(0..a_terms.len());
		let mut picked_term = String::from(a_terms[picked_term_i]);
		
		if verbose{
			println!("REMOVING {} FROM {}", picked_term, h_to_remove_from);
		}
		
		let mut index = algo.h_term_lists[&h_to_remove_from].iter().position(|x| *x == picked_term).unwrap();
		algo.h_term_lists.get_mut(&h_to_remove_from).map(|val| val.remove(index));
		
		algo.mult_algo.insert(h_to_remove_from.clone(), make_h(&algo.h_term_lists[&h_to_remove_from]));
		
	} else if mutation_type == 6{ // Remove one b-term from an h-term that has at least two b-terms
		let mut h_to_remove_from = String::from("");
		let mut valid = false;
		
		while !valid{
		
			h_to_remove_from.push('h');
			let mut h_no = rng.gen_range(1..num_terms+1);
			h_to_remove_from.push_str(h_no.to_string().as_str());
			
			while !algo.mult_algo.contains_key(&h_to_remove_from){

				h_to_remove_from = String::from("h");
				let mut h_no = rng.gen_range(1..num_terms+1);
				h_to_remove_from.push_str(h_no.to_string().as_str());
			}

			let mut num_b = 0;
		
			for term in 0..algo.h_term_lists[&h_to_remove_from].len(){
			
				if algo.h_term_lists[&h_to_remove_from][term].contains("b"){
					num_b +=1;	
				}
				
				if num_b > 1{
					valid = true;
				} else if !algo.solo_h_list_b.contains(&h_to_remove_from){
					algo.solo_h_list_b.push(h_to_remove_from.clone());
				}
				
				if algo.solo_h_list_b.len() == algo.h_term_lists.keys().len(){
					no_mutate = true;
					return (algo, num_terms, mutation_type, no_mutate, keys)
				}
			}
		}
	
		keys.push(h_to_remove_from.clone());
		
		let mut b_terms = [].to_vec();
		
		for x in 0..algo.h_term_lists[&h_to_remove_from].len(){
		
			if algo.h_term_lists[&h_to_remove_from][x].contains("b"){
				b_terms.push(&algo.h_term_lists[&h_to_remove_from][x]);
			}
		}
		
		let mut picked_term_i = rng.gen_range(0..b_terms.len());
		let mut picked_term = String::from(b_terms[picked_term_i]);
		
		if verbose{
			println!("REMOVING {} FROM {}", picked_term, h_to_remove_from);
		}
		
		let mut index = algo.h_term_lists[&h_to_remove_from].iter().position(|x| *x == picked_term).unwrap();
		algo.h_term_lists.get_mut(&h_to_remove_from).map(|val| val.remove(index));
		
		algo.mult_algo.insert(h_to_remove_from.clone(), make_h(&algo.h_term_lists[&h_to_remove_from]));	
	
	} else if mutation_type == 7{ // Add a random (+/-) h-term to a random c-term
		let mut h_no = rng.gen_range(1..num_terms+1);
		
		let mut rand = rng.gen_range(0..2);
		let mut h_to_add = String::from("");
		if rand == 0{ h_to_add.push('h') } else { h_to_add.push_str(" - h") }

		h_to_add.push_str(h_no.to_string().as_str());

		while !algo.mult_algo.contains_key(&h_to_add){
			h_no = rng.gen_range(1..num_terms+1);
			h_to_add = String::from("h");
			h_to_add.push_str(h_no.to_string().as_str());
		}
	
		let mut row = rng.gen_range(1..(mat_size.0)+1);
		let mut col = rng.gen_range(1..(mat_size.1)+1);
		let mut c_to_add_to = String::from("c");
		
		c_to_add_to.push_str(row.to_string().as_str());
		c_to_add_to.push_str(col.to_string().as_str());
		
		if algo.solo_c_list.contains(&c_to_add_to){ 
			let mut index = algo.solo_c_list.iter().position(|x| x == &c_to_add_to).unwrap();
			algo.solo_c_list.remove(index);
		}
		
		keys.push(c_to_add_to.clone());
		
		if verbose{
			println!("Adding {} to {}", h_to_add, c_to_add_to);
		}
		
		algo.c_term_lists.get_mut(&c_to_add_to).map(|val| val.push(h_to_add));
		algo.mult_algo.insert(c_to_add_to.clone(), make_c(&algo.c_term_lists[&c_to_add_to]));
		
	
	} else if mutation_type == 8{ // Remove one h-term from a c-term that has more than two h-terms
	let mut c_to_remove_from = String::from("c");
	let mut valid = false;
	
	while !valid{
	
		c_to_remove_from = String::from("c");
		let mut row = rng.gen_range(1..(mat_size.0)+1);
		let mut col = rng.gen_range(1..(mat_size.1)+1); 
		c_to_remove_from.push_str(row.to_string().as_str());
		c_to_remove_from.push_str(col.to_string().as_str());
		
		if algo.c_term_lists[&c_to_remove_from].len() > 2{
			valid = true;
		} else if !algo.solo_c_list.contains(&c_to_remove_from){
			algo.solo_c_list.push(c_to_remove_from.clone());
		}
		
		if algo.solo_c_list.len() == algo.c_term_lists.keys().len(){
			no_mutate = true;
			return (algo, num_terms, mutation_type, no_mutate, keys)
		}
	
	}
	
	keys.push(c_to_remove_from.clone());

	let mut picked_term_i = rng.gen_range(0..algo.c_term_lists[&c_to_remove_from].len());
	let mut picked_term = &algo.c_term_lists[&c_to_remove_from][picked_term_i];
	
	if verbose{
		println!("Removing {} from {}", picked_term, c_to_remove_from);
	}
	
	let mut index = algo.c_term_lists[&c_to_remove_from].iter().position(|x| x == picked_term).unwrap();
	algo.c_term_lists.get_mut(&c_to_remove_from).map(|val| val.remove(index));
	
	algo.mult_algo.insert(c_to_remove_from.clone(), make_c(&algo.c_term_lists[&c_to_remove_from]));
	
	}
	
	(algo, num_terms, mutation_type, no_mutate, keys)

}

// -----------------------------------------------------------------------------
// main: run the evolutionary search for a matrix multiplication algorithm
// -----------------------------------------------------------------------------
// Expects CLI: num_triples num_generations num_runs SA_constant num_mut
// Optional flags: "nocap", "pos", "SA". Creates matrix triples, a random initial
// algorithm, then for each generation: apply num_mut mutations, update int_maps,
// compute fitness; accept the mutant if fitness improves or (with SA) sometimes
// if it worsens. Writes progress and final algorithm to a file.
fn main() {
	let args: Vec<String> = env::args().collect();
	
	// Checks for enough arguments
	if args.len() < 3 {
		println!("Not enough arguments given, please try again.");
		exit(0x0100);
	}
	
	let num_triples = &args[1];
	let num_generations = &args[2];
	let num_runs = &args[3];
	let SA_const = &args[4];
	let num_mut = &args[5];
	

	for x in 0..num_runs.parse::<usize>().unwrap(){
		let mut filename = String::from(num_triples.to_string().as_str());
		filename.push_str("mat"); 

		let mut Matmult = MatMult {
			mat_triples: [].to_vec(),
			mat_size: (5,5),
			start_terms: 0, //Init to 125
			num_triples: 5,
			SMALL: 2,
			MEDIUM: 5,
			LARGE: 10,
			num_terms: 0, //Init to start_terms
			verbose: false,
			cells_priority: false,
			h_added: 0,	
			num_generations: 1000,	
			SA: false,
			num_mut: 1,
		};
		

		Matmult.start_terms = Matmult.mat_size.0 * Matmult.mat_size.0 * Matmult.mat_size.1; // Init start_terms
		Matmult.num_terms = Matmult.start_terms; 					    // Init num_terms
		Matmult.num_triples = num_triples.parse::<usize>().unwrap();						    // Init num_triples	
		Matmult.num_generations = num_generations.parse::<usize>().unwrap();						    // Init num_triples
		Matmult.num_mut = num_mut.parse::<usize>().unwrap();	
		
		init_mats(&mut Matmult.mat_triples, Matmult.num_triples, Matmult.mat_size); // build the matrix triples
		let mut algo = rand_algo(Matmult.MEDIUM, Matmult.num_terms, Matmult.mat_size);
		algo.num_h = Matmult.num_terms;

		if args.contains(&"nocap".to_string()){
			algo.h_cap = false;
			filename.push_str("nocap");
		} else {
			algo.h_cap = true;
			filename.push_str("cap");
		}
		
		if args.contains(&"pos".to_string()){
			algo.pos = true;
			filename.push_str("pos");
		} else {
			algo.pos = false;
			filename.push_str("nopos");
		}
		
		filename.push_str(num_mut.to_string().as_str());
		filename.push_str("mut");
		
		
		if args.contains(&"SA".to_string()){
			Matmult.SA = true;
			filename.push_str("SA");
			let mut SA_num = SA_const.parse::<f64>().unwrap();;
			let mut SA_new: i32 = SA_num.log(10.0).round() as i32;
			SA_new = (SA_new * -1) - 1; 
			println!("LOOK HERE {}", SA_new);
			filename.push_str(SA_new.to_string().as_str());
			filename.push_str("-");
		}
		
		let run_no = x+1;
		filename.push_str(run_no.to_string().as_str());
		
		let mut new_algo = algo.clone();
		let mut fitness: (f64, f64) = (0.0,0.0);
		init_maps(&mut algo, &Matmu2lt, &Matmult.mat_triples, Matmult.num_terms, Matmult.MEDIUM, Matmult.mat_size);
		init_maps(&mut new_algo, &Matmult, &Matmult.mat_triples, Matmult.num_terms, Matmult.MEDIUM, Matmult.mat_size);

		fitness = get_fitness(&mut algo, &Matmult, &Matmult.mat_triples, Matmult.num_terms, Matmult.MEDIUM, Matmult.mat_size);
		let n = [].to_vec();
		update_maps(&mut algo, &Matmult.mat_triples, Matmult.num_terms, Matmult.MEDIUM, Matmult.mat_size, n.clone());
		//update_maps(&mut new_algo, &Matmult.mat_triples, Matmult.num_terms, Matmult.MEDIUM, Matmult.mat_size, n.clone());
		
		//print_int_maps(&algo, Matmult.num_terms, Matmult.mat_size);
		//print_mats(&Matmult.mat_triples[0]);
		//print_mat3_algo(&algo);
		
		let mut file = File::create(filename.clone());
		
		let mut data_file = OpenOptions::new()
		.append(true)
		.open(filename)
		.expect("cannot open file");
		
		// --- Evolution loop: mutate, re-evaluate, accept/reject by fitness (and optionally SA) ---
		for x in 0..Matmult.num_generations{

			let current_gen: f64 = x as f64;
			let max_gen: f64 = Matmult.num_generations as f64;
			let constant: f64 = SA_const.parse::<f64>().unwrap();;
			let power: f64 = (current_gen/max_gen);
			algo.temp = constant.powf(power); // Can be changed

			if x % 1000 == 0 || current_gen == max_gen-1.0{
				println!("Gen {}, const {}, power {}, temp {}", x, constant, power, algo.temp);
				println!("Generation {}, fitness difference: {}, fitness_cells: {}, num h: {}, temp: {}", x, fitness.0, fitness.1, algo.num_h, algo.temp); 
				println!("mat size: {}, num triples: {}, num terms: {}, num generations: {}, num mutations: {}, SA: {}, SA constant: {}, SA power: {}", Matmult.mat_size.0, Matmult.num_triples, Matmult.num_terms, Matmult.num_generations, Matmult.num_mut, Matmult.SA, SA_const, power);
				
				data_file.write(x.to_string().as_bytes());
				data_file.write(",".as_bytes());
				data_file.write(fitness.0.to_string().as_bytes());
				data_file.write(",".as_bytes());
				data_file.write(fitness.1.to_string().as_bytes());
				data_file.write("\n".as_bytes());
			}
			
			// performs mutations
			for x in 0..Matmult.num_mut{
			
				let res = mutate(&mut new_algo, Matmult.MEDIUM, Matmult.num_terms, Matmult.mat_size, Matmult.num_triples, false);
				Matmult.num_terms = res.1.clone();
				let keys = res.4.clone();
				
				update_maps(&mut new_algo, &Matmult.mat_triples, Matmult.num_terms, Matmult.MEDIUM, Matmult.mat_size, keys);

				//thread::sleep_ms(1000);
			}
			
			let new_fitness = get_fitness(&mut new_algo, &Matmult, &Matmult.mat_triples, Matmult.num_terms, Matmult.MEDIUM, Matmult.mat_size);

			// Selection: accept new_algo if better; if SA, sometimes accept worse with probability temp
			if !Matmult.SA{
				if new_fitness.0 == 0.0{
					println!("solved!");
					data_file.write("solved! \n".as_bytes());
					fitness = new_fitness;
					break;
				} else if new_fitness.0 < fitness.0{
					//println!("LESS, NEW FITNESS: {}", new_fitness.0);
					algo = new_algo.clone();
					fitness = new_fitness
				} else if new_fitness.0 == fitness.0{
					//println!("SAME");
					if !algo.pos{
						algo = new_algo.clone();
						fitness = new_fitness
					} else {
						new_algo = algo.clone();
					}
				} else if new_fitness.0 > fitness.0{
					//println!("MORE, FITNESS: {}", new_fitness.0);
					new_algo = algo.clone();
				}		
			} else if Matmult.SA {
				if new_fitness.0 == 0.0{
					println!("solved!");
					data_file.write("solved! \n".as_bytes());
					fitness = new_fitness;
					break;
				} else if new_fitness.0 < fitness.0{
					//println!("LESS, NEW FITNESS: {}", new_fitness.0);
					algo = new_algo.clone();
					fitness = new_fitness
				} else if new_fitness.0 == fitness.0{
					//println!("SAME");
					if !algo.pos{
						algo = new_algo.clone();
						fitness = new_fitness
					}
				} else if new_fitness.0 > fitness.0{
					//println!("MORE, FITNESS: {}", new_fitness.0);
					let mut rng = rand::thread_rng();
					let randfloat = rng.gen_range(0.0..1.0);
					if randfloat < algo.temp {
						algo = new_algo.clone();
						fitness = new_fitness
					}
					new_algo = algo.clone();
				}	
			}
		}
		
		
		// print_algo(&algo, Matmult.num_terms, Matmult.mat_size, data_file);
		println!("\n=== TRUE C Matrix ===");
		for row in 0..5 {
			println!("{} {} {} {} {}", 
				Matmult.mat_triples[0].2.get((row,0)),
				Matmult.mat_triples[0].2.get((row,1)),
				Matmult.mat_triples[0].2.get((row,2)),
				Matmult.mat_triples[0].2.get((row,3)),
				Matmult.mat_triples[0].2.get((row,4)));
		}
		
		// Print computed C matrix
		println!("\n=== COMPUTED C Matrix ===");
		print_mat3_algo(&algo);
				
		println!("fit_diff {}", fitness.0);
		println!("fit_cells {}", fitness.1);
	}
	
	let mut rng = rand::thread_rng();
	
	let num: f64 = 0.01;
	let res: f64 = num.log(10.0).round();
	
	println!("divided: {}", res);
	/*
	
	let mut file = File::create("data.txt");
	
	let mut data_file = OpenOptions::new()
        .append(true)
        .open("data.txt")
        .expect("cannot open file");

    	// Write to a file
    	for x in 0..6{
	    	data_file
		.write("I am learning Rust!".as_bytes())
		.expect("write failed");
	}
    
	*/
}
