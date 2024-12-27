#ifndef IMAGE_H
#define IMAGE_H

enum class Border : int
{
    Clamp = 0,  // aaaaaa|abcdefgh|hhhhhhh
    Wrap,       // cdefgh|abcdefgh|abcdefg
    Reflect,    // gfedcb|abcdefgh|gfedcba
    Mirror,     // fedcba|abcdefgh|hgfedcb

    Constant,   // use constant value provided

    Last = Constant      // Last useful value in the border enum
};

enum Result
{
    Ok = 0,				//!< Operation completed successfully.
    InvalidArgument,	//!< Operation was passed an invalid argument.
    InvalidOperation,   //!< Operation not permitted.
    FileIOFailure,		//!< Operation failed performing file I/O.
    OutOfMemory,		//!< Operation failed after running out of memory/resource.
    KernelLaunchFailed,	//!< GPU program failed to run.
    MemCopyFailed,		//!< A memory copy operation failed.
    KernelCompilationFailed,	//!< A kernel program compilation failed.
    NotImplemented,		//!< Function/method not implemented.
    Unknown1				//!< Unknown error has occured.
};

#endif