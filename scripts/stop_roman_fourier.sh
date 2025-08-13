### unlink emulator likelihoods from the usual cocoa likelihood
if [ -d "$ROOTDIR/projects/roman_fourier/emulator/likelihood" ]; then
	rm $ROOTDIR/projects/roman_fourier/likelihood/*_emu.py
    rm $ROOTDIR/projects/roman_fourier/likelihood/*_emu.yaml
fi

unset SPDLOG_LEVEL
