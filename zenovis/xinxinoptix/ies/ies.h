/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef __UTIL_IES_H__
#define __UTIL_IES_H__

#include <string>
#include <vector>

namespace blender {
using namespace std;

class IESFile {
 public:
  IESFile() {}
  ~IESFile();

  int packed_size();
  void pack(float *data);

  float coneAngle() {
		if (v_angles.empty()) 
      return 0.0f; 

    return max(v_angles.front(), v_angles.back());
  }

  bool load(const string &ies);
  void clear();

 protected:
  bool parse(const string &ies);
  bool process();
  bool process_type_b();
  bool process_type_c();

  /* The brightness distribution is stored in spherical coordinates.
   * The horizontal angles correspond to theta in the regular notation
   * and always span the full range from 0° to 360°.
   * The vertical angles correspond to phi and always start at 0°. */
  std::vector<float> v_angles, h_angles;
  /* The actual values are stored here, with every entry storing the values
   * of one horizontal segment. */
  std::vector<std::vector<float>> intensity;

  /* Types of angle representation in IES files. Currently, only B and C are supported. */
  enum IESType { TYPE_A = 3, TYPE_B = 2, TYPE_C = 1 } type;
};

} // namespace end

#endif /* __UTIL_IES_H__ */
