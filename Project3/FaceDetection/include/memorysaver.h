// Project-level ArduCAM sensor selector.
// PlatformIO searches include/ before library paths, so this file takes
// precedence over the library's own memorysaver.h and activates only the
// OV2640 Mini 2MP Plus driver, keeping compiled code size minimal.

#ifndef MEMORYSAVER_H
#define MEMORYSAVER_H

#ifndef OV2640_MINI_2MP_PLUS
#define OV2640_MINI_2MP_PLUS
#endif

#endif // MEMORYSAVER_H
