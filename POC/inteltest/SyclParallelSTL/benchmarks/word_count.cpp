/* Copyright (c) 2015-2018 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

*/

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>

#include <experimental/algorithm>
#include <sycl/execution_policy>

#include "benchmark.h"

using namespace sycl::helpers;

// This example computes the number of words in a text sample
// with a single call to thrust::inner_product.  The algorithm
// counts the number of characters which start a new word, i.e.
// the number of characters where input[i] is an alphabetical
// character and input[i-1] is not an alphabetical character.

// determines whether the character is alphabetical
bool is_alpha(const char c) { return (c >= 'A' && c <= 'z'); }

// determines whether the right character begins a new word
bool is_word_start(char left, char right) {
  return !is_alpha(left) && is_alpha(right);
}

template <class InputIterator>
int word_count(InputIterator first1, InputIterator last1,
               InputIterator first2) {
  // check for empty string
  if (std::distance(first1, last1) < 1) return 0;

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class WordCountAlgorithm> snp(q);
  // compute the number characters that start a new word
  int wc = std::experimental::parallel::inner_product(
      snp, first1, last1,                      // sequence of left characters
      first2,                                  // sequence of right characters
      0,                                       // initialize sum to 0
      [](int s1, int s2) { return s1 + s2; },  // sum values together
      [](char s1, char s2) {
        if (is_word_start(s1, s2)) {
          return 1;
        } else {
          return 0;
        }
      });  // how to compare the left and right characters

  // if the first character is alphabetical, then it also begins a word
  if (is_alpha(*first1)) {
    wc++;
  }

  return wc;
}

int main() {
  // Paragraph from 'Don Quixote' by Miguel de Cervantes
  // http://www.learnlibrary.com/don-quixote/index.htm
  std::string raw_input =
      " In a village of La Mancha, the name of which I have no desire to call "
      "to mind, there lived not long since one of those gentlemen that keep a "
      "lance in the lance-rack, an old buckler, a lean hack, and a greyhound "
      "for coursing. An olla of rather more beef than mutton, a salad on most "
      "nights, scraps on Saturdays, lentils on Fridays, and a pigeon or so "
      "extra on Sundays, made away with three-quarters of his income. The rest "
      "of it went in a doublet of fine cloth and velvet breeches and shoes to "
      "match for holidays, while on week-days he made a brave figure in his "
      "best homespun. He had in his house a housekeeper past forty, a niece "
      "under twenty, and a lad for the field and market-place, who used to "
      "saddle the hack as well as handle the bill-hook. The age of this "
      "gentleman of ours was bordering on fifty; he was of a hardy habit, "
      "spare, gaunt-featured, a very early riser and a great sportsman. They "
      "will have it his surname was Quixada or Quesada (for here there is some "
      "difference of opinion among the authors who write on the subject), "
      "although from reasonable conjectures it seems plain that he was called "
      "Quexana. This, however, is of but little importance to our tale; it "
      "will be enough not to stray a hair's breadth from the truth in the "
      "telling of it.";

  std::cout << "Text sample:" << std::endl;
  std::cout << raw_input << std::endl;

  // count words
  int wc =
      word_count(begin(raw_input), end(raw_input) - 1, begin(raw_input) + 1);

  std::cout << "Text sample contains " << wc << " words" << std::endl;

  return 0;
}
