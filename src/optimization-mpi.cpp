/*
  Branch and bound algorithm to find the minimum of continuous binary 
  functions using interval arithmetic.

  Sequential version

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <mpi.h>
#include "interval.h"
#include "functions.h"
#include "minimizer.h"

using namespace std;


// Split a 2D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval& x, const interval& y,
	       interval &xl, interval& xr, interval& yl, interval& yr)
{
  double xm = x.mid();
  double ym = y.mid();
  xl = interval(x.left(),xm);
  xr = interval(xm,x.right());
  yl = interval(y.left(),ym);
  yr = interval(ym,y.right());
}

// Branch-and-bound minimization algorithm
void minimize(itvfun f,  // Function to minimize
	      const interval& x, // Current bounds for 1st dimension
	      const interval& y, // Current bounds for 2nd dimension
	      double threshold,  // Threshold at which we should stop splitting
	      double& min_ub,  // Current minimum upper bound
	      minimizer_list& ml) // List of current minimizers
{
  interval fxy = f(x,y);
  
  if (fxy.left() > min_ub) { // Current box cannot contain minimum?
    return ;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum?
    min_ub = fxy.right();
    // Discarding all saved boxes whose minimum lower bound is 
    // greater than the new minimum upper bound
    auto discard_begin = ml.lower_bound(minimizer{0,0,min_ub,0});
    ml.erase(discard_begin,ml.end());
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) { 
    // We have potentially a new minimizer
    ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
    return ;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x,y,xl,xr,yl,yr);

  minimize(f,xl,yl,threshold,min_ub,ml);
  minimize(f,xl,yr,threshold,min_ub,ml);
  minimize(f,xr,yl,threshold,min_ub,ml);
  minimize(f,xr,yr,threshold,min_ub,ml);
}

// Branch-and-bound minimization algorithm
void minimize_first(itvfun f,  // Function to minimize
	      const interval& x, // Current bounds for 1st dimension
	      const interval& y, // Current bounds for 2nd dimension
	      double threshold,  // Threshold at which we should stop splitting
	      double& min_ub,  // Current minimum upper bound
	      minimizer_list& ml,// List of current minimizers
	      int rank) 
{
	
	
	

  interval fxy = f(x,y);
  
  if (fxy.left() > min_ub) { // Current box cannot contain minimum?
    return ;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum?
    min_ub = fxy.right();
    // Discarding all saved boxes whose minimum lower bound is 
    // greater than the new minimum upper bound
    auto discard_begin = ml.lower_bound(minimizer{0,0,min_ub,0});
    ml.erase(discard_begin,ml.end());
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) { 
    // We have potentially a new minimizer
    ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
    return ;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x,y,xl,xr,yl,yr);
	
	if ( rank == 0 )
	{
		minimize(f,xl,yl,threshold,min_ub,ml);
	}
	else if ( rank == 1 )
	{
		minimize(f,xl,yr,threshold,min_ub,ml);
	}
	else if ( rank == 2 )
	{
		minimize(f,xr,yl,threshold,min_ub,ml);
	}
	else if ( rank == 3 )
	{
		minimize(f,xr,yr,threshold,min_ub,ml);
	}
  
  
  
  
}


int main(int argc, char **argv)
{
	int nbProc;
	int rank;
	
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &nbProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
  cout.precision(16);
  // By default, the currently known upper bound for the minimizer is +oo
  double min_ub = numeric_limits<double>::infinity();
  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is 
  // greater than the new current upper bound
  minimizer_list minimums;
  // Threshold at which we should stop splitting a box
  double precision;

  // Name of the function to optimize
  string choice_fun;

  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;
  
  bool good_choice;

  // Asking the user for the name of the function to optimize
  if ( rank == 0 )
	{
		do {
		  good_choice = true;

		  cout << "Which function to optimize?\n";
		  cout << "Possible choices: ";
		  for (auto fname : functions) {
		    cout << fname.first << " ";
		  }
		  cout << endl;
		  cin >> choice_fun;
		  
		  try {
		    fun = functions.at(choice_fun);
		  } catch (out_of_range) {
		    cerr << "Bad choice" << endl;
		    good_choice = false;
		  }
		} while(!good_choice);
	
		// Asking for the threshold below which a box is not split further
		cout << "Precision? ";
		cin >> precision;
  
  	minimize_first(fun.f,fun.x,fun.y,precision,min_ub,minimums, rank);
  
  }
  
  // Displaying all potential minimizers
  copy(minimums.begin(),minimums.end(),
       ostream_iterator<minimizer>(cout,"\n"));    
  cout << "Number of minimizers: " << minimums.size() << endl;
  cout << "Upper bound for minimum: " << min_ub << endl;
  
  
  
  MPI_Finalize();
  
  return 0;
}
