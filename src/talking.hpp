/*******************************************************************************
 * This file is part of MEGA++.
 * Copyright (c) 2023 Will Roper (w.roper@sussex.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * This header file contains prototypes related to "talking" to the user.
 ******************************************************************************/

// Standard includes
#include <iostream>
#include <string>

// Local includes
#include "version.h"

using namespace std;

#ifndef TALKING_H_
#define TALKING_H_

std::string padString(const std::string &input, std::size_t length) {

  /* Set up the result. */
  std::string result = input;

  /* Loop until the desired length is reached. */
  while (result.length() < length) {
    result += " ";
  }
  return result;
}

/**
 * @brief Prints a greeting message to the standard output containing code
 * version and revision number
 *
 * This was constructed using the 'figlet' tool and the 'slant' font. The
 * lower-bar of the f is then lengthened.
 *
 * @param fof Is this for the FOF greeting?
 */
void say_hello() {

  string string1 =
      R"( ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ )";
  string string2 =
      R"(||P |||A |||R |||E |||N |||T |||- |||G |||R |||I |||D |||D |||E |||R ||)";
  string string3 =
      R"(||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__||)";
  string string4 =
      R"(|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|)";

  cout << endl;
  cout << string1 << endl;
  cout << string2 << endl;
  cout << string3 << endl;
  cout << string4 << endl;
  cout << endl;

  /* Report some information about the MEGA version being run. */
  int nPad = 30;
  cout << padString(string(" Version : "), nPad) << PROJECT_VERSION_MAJOR << "."
       << PROJECT_VERSION_MINOR << "." << PROJECT_VERSION_PATCH << endl;

  cout << endl;

  cout << " Git:" << endl
       << padString(string(" On branch: "), nPad) << GIT_BRANCH << endl
       << padString(string(" Using revision: "), nPad) << GIT_REVISION << endl
       << padString(string(" Last updated: "), nPad) << GIT_DATE << endl;
  // printf(" Webpage : %s\n\n", PACKAGE_URL);
  // printf(" Config. options: %s\n\n", configuration_options());

  cout << endl;

  cout << padString(string(" Compiler: "), nPad) << COMPILER_INFO << endl;
  cout << padString(string(" CFLAGS: "), nPad) << CFLAGS_INFO << endl;

  cout << endl;

  cout << padString(string(" HDF5 library version: "), nPad) << HDF5_VERSION
       << endl;
  // #ifdef HAVE_FFTW
  //     printf(" FFTW library version     : %s\n", fftw3_version());
  // #endif
  // #ifdef HAVE_LIBGSL
  //     printf(" GSL library version      : %s\n", libgsl_version());
  // #endif
  // #ifdef WITH_MPI
  //     printf(" MPI library version      : %s\n", mpi_version());
  // #endif
  printf("\n");
}

#endif // TALKING_H_
