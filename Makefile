CXX=g++
CXXFLAGS = -std=c++20 -pthread -MD -MP
LDFLAGS :=

SRCDIR  = ./src
OBJSDIR = ./build
DEPDIR	= ./include
BINDIR = ./bin
OUTPUT = $(BINDIR)/engine

#  Release version is compiled with optimisation flag
ifeq ($(RELEASE), 1)
	CXXFLAGS += -Ofast
endif

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g
endif

ifeq ($(LEARNING), 1)
	CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0 -DLEARNING=1
	LDFLAGS += -ltorch -ltorch_cpu -lc10 -lgomp
	INCLUDES += -I/usr/include/torch/csrc/api/include
endif

# Find all subdirectories
INCLUDES += $(shell find $(SRCDIR) -type d | sed s/^/-I/) -I$(DEPDIR)

# Get all source files from source directory
SOURCES = $(shell find $(SRCDIR) -type f -name '*.cpp')

# Generate all objects from source file
OBJECTS  = $(SOURCES:$(SRCDIR)%.cpp=$(OBJSDIR)%.o)

# Main rule
$(OUTPUT): $(BINDIR) $(OBJSDIR) $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $(OUTPUT)

# Make each object
$(OBJECTS): $(OBJSDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Generate dependency files for each object
# This makes sure that not everything is recompiled if headers are modified
-include $(SOURCES:$(SRCDIR)%.cpp=$(OBJSDIR)%.d)

# Recusively copy directory structure of src/ to build/ without files
$(OBJSDIR):
	rsync -av -f"+ */" -f"- *" $(SRCDIR)/ $(OBJSDIR)

$(BINDIR):
	mkdir $(BINDIR)

all:
	make

release:
	make RELEASE=1 LEARNING=1

debug:
	make DEBUG=1

clean:
	rm -f $(OUTPUT)
	rm -rf $(OBJSDIR)
