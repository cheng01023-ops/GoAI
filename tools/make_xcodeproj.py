#!/usr/bin/env python3
"""Generate GoAI.xcodeproj (a hand-rolled pbxproj, no external tools needed)."""
import hashlib, os

def uid(key):
    return hashlib.md5(key.encode()).hexdigest()[:24].upper()

PROJ = uid('project')

SRC_FILES   = ['src/board.c', 'src/compat.c', 'src/net.c', 'src/mcts.c', 'src/remote.c', 'src/gplay.c', 'src/train.c', 'src/apps.c', 'src/main.c']
INC_FILES   = ['include/board.h', 'include/compat.h', 'include/rand.h', 'include/net.h', 'include/mcts.h',
               'include/remote.h', 'include/gplay.h',
               'include/train.h', 'include/apps.h']
TEST_FILES  = ['tests/test_main.c', 'tests/test_board.c', 'tests/test_rng.c', 'tests/test_net.c', 'tests/test_mcts.c']
TEST_SHARED = ['src/board.c', 'src/compat.c', 'src/net.c', 'src/mcts.c']
BOARD_FILES = ['src/goboard.c'] + TEST_SHARED

def ref(path):        return uid('ref:' + path)
def bf(target, path): return uid('bf:%s:%s' % (target, path))

objects = []
def add(s):
    objects.append(s)

# ---------- file references ----------
filetype = {'.c': 'sourcecode.c.c', '.h': 'sourcecode.c.h'}
for p in SRC_FILES + INC_FILES + TEST_FILES + ['src/goboard.c']:
    name = os.path.basename(p)
    add('\t\t%s /* %s */ = {isa = PBXFileReference; lastKnownFileType = %s; path = "%s"; sourceTree = "<group>"; };'
        % (ref(p), name, filetype[os.path.splitext(p)[1]], name))

# ---------- build files ----------
for p in SRC_FILES:
    add('\t\t%s /* %s in Sources */ = {isa = PBXBuildFile; fileRef = %s /* %s */; };'
        % (bf('GoAI', p), os.path.basename(p), ref(p), os.path.basename(p)))
for p in TEST_FILES + TEST_SHARED:
    add('\t\t%s /* %s in Sources */ = {isa = PBXBuildFile; fileRef = %s /* %s */; };'
        % (bf('Tests', p), os.path.basename(p), ref(p), os.path.basename(p)))
for p in BOARD_FILES:
    add('\t\t%s /* %s in Sources */ = {isa = PBXBuildFile; fileRef = %s /* %s */; };'
        % (bf('Board', p), os.path.basename(p), ref(p), os.path.basename(p)))

# ---------- groups ----------
def group(gid, name, children, path=None, comment=None):
    kids = ''.join('\n\t\t\t\t%s /* %s */,' % (c, os.path.basename(c_path.get(c, c))) for c in children)
    pathline = '\n\t\t\tpath = %s;' % path if path else ''
    add('\t\t%s /* %s */ = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (%s\n\t\t\t);%s\n\t\t\tsourceTree = "<group>";\n\t\t};'
        % (gid, comment or name, kids, pathline))

c_path = {ref(p): p for p in SRC_FILES + INC_FILES + TEST_FILES + ['src/goboard.c']}
SRC_GROUP  = uid('group:src')
INC_GROUP  = uid('group:include')
TEST_GROUP = uid('group:tests')
PROD_GROUP = uid('group:products')
MAIN_GROUP = uid('group:main')
group(SRC_GROUP,  'src',     [ref(p) for p in SRC_FILES + ['src/goboard.c']], path='src')
group(INC_GROUP,  'include', [ref(p) for p in INC_FILES],  path='include')
group(TEST_GROUP, 'tests',   [ref(p) for p in TEST_FILES], path='tests')
group(PROD_GROUP, 'Products', [uid('product:GoAI'), uid('product:Tests'), uid('product:Board')], comment='Products')
add('\t\t%s /* Products */ = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (\n\t\t\t\t%s /* GoAI */,\n\t\t\t\t%s /* GoAITests */,\n\t\t\t);\n\t\t\tname = Products;\n\t\t\tsourceTree = "<group>";\n\t\t};'
    % (PROD_GROUP, uid('product:GoAI'), uid('product:Tests')))
add('\t\t%s = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (\n\t\t\t\t%s /* src */,\n\t\t\t\t%s /* include */,\n\t\t\t\t%s /* tests */,\n\t\t\t\t%s /* Products */,\n\t\t\t);\n\t\t\tsourceTree = "<group>";\n\t\t};'
    % (MAIN_GROUP, SRC_GROUP, INC_GROUP, TEST_GROUP, PROD_GROUP))

# ---------- products ----------
for tgt, name in (('GoAI', 'GoAI'), ('Tests', 'GoAITests'), ('Board', 'GoBoard')):
    add('\t\t%s /* %s */ = {isa = PBXFileReference; explicitFileType = "compiled.mach-o.executable"; includeInIndex = 0; path = %s; sourceTree = BUILT_PRODUCTS_DIR; };'
        % (uid('product:' + tgt), name, name))

# ---------- build phases ----------
def sources_phase(tgt, files):
    ids = ''.join('\n\t\t\t\t%s /* %s in Sources */,' % (bf(tgt, p), os.path.basename(p)) for p in files)
    add('\t\t%s /* Sources */ = {\n\t\t\tisa = PBXSourcesBuildPhase;\n\t\t\tbuildActionMask = 2147483647;\n\t\t\tfiles = (%s\n\t\t\t);\n\t\t\trunOnlyForDeploymentPostprocessing = 0;\n\t\t};'
        % (uid('sources:' + tgt), ids))
add('\t\t%s /* Frameworks */ = {\n\t\t\tisa = PBXFrameworksBuildPhase;\n\t\t\tbuildActionMask = 2147483647;\n\t\t\tfiles = (\n\t\t\t);\n\t\t\trunOnlyForDeploymentPostprocessing = 0;\n\t\t};' % uid('frameworks:GoAI'))
add('\t\t%s /* Frameworks */ = {\n\t\t\tisa = PBXFrameworksBuildPhase;\n\t\t\tbuildActionMask = 2147483647;\n\t\t\tfiles = (\n\t\t\t);\n\t\t\trunOnlyForDeploymentPostprocessing = 0;\n\t\t};' % uid('frameworks:Tests'))
add('\t\t%s /* Frameworks */ = {isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (\n\t\t\t);\n\t\t\trunOnlyForDeploymentPostprocessing = 0;\n\t\t};' % uid('frameworks:Board'))
sources_phase('GoAI', SRC_FILES)
sources_phase('Tests', TEST_FILES + TEST_SHARED)
sources_phase('Board', BOARD_FILES)

# ---------- build configurations ----------
COMMON = '''\t\t\t\tALWAYS_SEARCH_USER_PATHS = NO;
\t\t\t\tCLANG_ENABLE_MODULES = YES;
\t\t\t\tCLANG_ENABLE_OBJC_ARC = YES;
\t\t\t\tGCC_C_LANGUAGE_STANDARD = c11;
\t\t\t\tGCC_NO_COMMON_BLOCKS = YES;
\t\t\t\tGCC_WARN_UNINITIALIZED_AUTOS = YES;
\t\t\t\tGCC_WARN_UNUSED_VARIABLE = YES;
\t\t\t\tHEADER_SEARCH_PATHS = "$(SRCROOT)/include";
\t\t\t\tMACOSX_DEPLOYMENT_TARGET = 12.0;
\t\t\t\tSDKROOT = macosx;'''

def config(cid, name, settings, base=False):
    add('\t\t%s /* %s */ = {\n\t\t\tisa = XCBuildConfiguration;\n\t\t\tbuildSettings = {\n%s\n\t\t\t};\n\t\t\tname = %s;\n\t\t};'
        % (cid, name, settings, name))

config(uid('cfg:proj:Debug'), 'Debug', COMMON + '''
\t\t\t\tDEBUG_INFORMATION_FORMAT = dwarf;
\t\t\t\tENABLE_TESTABILITY = YES;
\t\t\t\tGCC_OPTIMIZATION_LEVEL = 0;
\t\t\t\tGCC_PREPROCESSOR_DEFINITIONS = (
\t\t\t\t\t"DEBUG=1",
\t\t\t\t\t"$(inherited)",
\t\t\t\t);
\t\t\t\tONLY_ACTIVE_ARCH = YES;''')
config(uid('cfg:proj:Release'), 'Release', COMMON + '''
\t\t\t\tDEBUG_INFORMATION_FORMAT = "dwarf-with-dsym";
\t\t\t\tENABLE_NS_ASSERTIONS = NO;
\t\t\t\tGCC_OPTIMIZATION_LEVEL = 3;''')
config(uid('cfg:GoAI:Debug'),   'Debug',   '\t\t\t\tPRODUCT_NAME = GoAI;')
config(uid('cfg:GoAI:Release'), 'Release', '\t\t\t\tPRODUCT_NAME = GoAI;')
config(uid('cfg:Tests:Debug'),   'Debug',   '\t\t\t\tPRODUCT_NAME = GoAITests;')
config(uid('cfg:Tests:Release'), 'Release', '\t\t\t\tPRODUCT_NAME = GoAITests;')
config(uid('cfg:Board:Debug'),   'Debug',   '\t\t\t\tPRODUCT_NAME = GoBoard;\n\t\t\t\tOTHER_LDFLAGS = "-lncurses";')
config(uid('cfg:Board:Release'), 'Release', '\t\t\t\tPRODUCT_NAME = GoBoard;\n\t\t\t\tOTHER_LDFLAGS = "-lncurses";')

for key in ('proj', 'GoAI', 'Tests', 'Board'):
    add('\t\t%s /* Build configuration list for PBXProject/NativeTarget */ = {\n\t\t\tisa = XCConfigurationList;\n\t\t\tbuildConfigurations = (\n\t\t\t\t%s /* Debug */,\n\t\t\t\t%s /* Release */,\n\t\t\t);\n\t\t\tdefaultConfigurationIsVisible = 0;\n\t\t\tdefaultConfigurationName = Release;\n\t\t};'
        % (uid('cfglist:' + key), uid('cfg:%s:Debug' % key), uid('cfg:%s:Release' % key)))

# ---------- targets ----------
for tgt, name in (('GoAI', 'GoAI'), ('Tests', 'GoAITests'), ('Board', 'GoBoard')):
    add('\t\t%s /* %s */ = {\n\t\t\tisa = PBXNativeTarget;\n\t\t\tbuildConfigurationList = %s;\n\t\t\tbuildPhases = (\n\t\t\t\t%s /* Sources */,\n\t\t\t\t%s /* Frameworks */,\n\t\t\t);\n\t\t\tbuildRules = (\n\t\t\t);\n\t\t\tdependencies = (\n\t\t\t);\n\t\t\tname = %s;\n\t\t\tproductName = %s;\n\t\t\tproductReference = %s /* %s */;\n\t\t\tproductType = "com.apple.product-type.tool";\n\t\t};'
        % (uid('target:' + tgt), name, uid('cfglist:' + tgt), uid('sources:' + tgt), uid('frameworks:' + tgt),
           name, name, uid('product:' + tgt), name))

add('''\t\t%s /* Project object */ = {
\t\t\tisa = PBXProject;
\t\t\tattributes = {
\t\t\t\tBuildIndependentTargetsInParallel = 1;
\t\t\t\tLastUpgradeCheck = 1500;
\t\t\t};
\t\t\tbuildConfigurationList = %s;
\t\t\tcompatibilityVersion = "Xcode 14.0";
\t\t\tdevelopmentRegion = en;
\t\t\thasScannedForEncodings = 0;
\t\t\tknownRegions = (
\t\t\t\ten,
\t\t\t\tBase,
\t\t\t);
\t\t\tmainGroup = %s;
\t\t\tproductRefGroup = %s /* Products */;
\t\t\tprojectDirPath = "";
\t\t\tprojectRoot = "";
\t\t\ttargets = (
\t\t\t\t%s /* GoAI */,
\t\t\t\t%s /* GoAITests */,
\t\t\t);
\t\t};''' % (PROJ, uid('cfglist:proj'), MAIN_GROUP, PROD_GROUP, uid('target:GoAI'), uid('target:Tests')))

content = '''// !$*UTF8*$!
{
\tarchiveVersion = 1;
\tclasses = {
\t};
\tobjectVersion = 54;
\tobjects = {

%s
\t};
\trootObject = %s /* Project object */;
}
''' % ('\n\n'.join(objects), PROJ)

os.makedirs('GoAI.xcodeproj/xcshareddata/xcschemes', exist_ok=True)
# --- 把 GoBoard 追加进 Products 组与 targets 列表（直接对生成文本做后处理，避免 % 格式化踩坑） ---
content = content.replace(
    '/* GoAITests */,\n\t\t\t);\n\t\t\tname = Products;',
    '/* GoAITests */,\n\t\t\t\t%s /* GoBoard */,\n\t\t\t);\n\t\t\tname = Products;' % uid('product:Board'))
content = content.replace(
    '/* GoAITests */,\n\t\t\t);\n\t\t};',
    '/* GoAITests */,\n\t\t\t\t%s /* GoBoard */,\n\t\t\t);\n\t\t};' % uid('target:Board'))
open('GoAI.xcodeproj/project.pbxproj', 'w').write(content)

SCHEME = '''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion = "1500" version = "1.7">
   <BuildAction parallelizeBuildables = "YES" buildImplicitDependencies = "YES">
      <BuildActionEntries>
         <BuildActionEntry buildForTesting = "YES" buildForRunning = "YES" buildForProfiling = "YES" buildForArchiving = "YES" buildForAnalyzing = "YES">
            <BuildableReference
               BuildableIdentifier = "primary"
               BlueprintIdentifier = "%s"
               BuildableName = "%s"
               BlueprintName = "%s"
               ReferencedContainer = "container:GoAI.xcodeproj">
            </BuildableReference>
         </BuildActionEntry>
      </BuildActionEntries>
   </BuildAction>
   <TestAction buildConfiguration = "Debug" selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier = "Xcode.DebuggerFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv = "YES">
      <Testables>
      </Testables>
   </TestAction>
   <LaunchAction buildConfiguration = "Debug" selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier = "Xcode.DebuggerFoundation.Launcher.LLDB" launchStyle = "0" useCustomWorkingDirectory = "YES" customWorkingDirectory = "$(SRCROOT)" ignoresPersistentStateOnLaunch = "NO" debugDocumentVersioning = "YES" debugServiceExtension = "internal" allowLocationSimulation = "YES">
      <BuildableProductRunnable runnableDebuggingMode = "0">
         <BuildableReference
            BuildableIdentifier = "primary"
            BlueprintIdentifier = "%s"
            BuildableName = "%s"
            BlueprintName = "%s"
            ReferencedContainer = "container:GoAI.xcodeproj">
         </BuildableReference>
      </BuildableProductRunnable>
   </LaunchAction>
   <ProfileAction buildConfiguration = "Release" shouldUseLaunchSchemeArgsEnv = "YES" savedToolIdentifier = "" useCustomWorkingDirectory = "YES" customWorkingDirectory = "$(SRCROOT)" debugDocumentVersioning = "YES">
      <BuildableProductRunnable runnableDebuggingMode = "0">
         <BuildableReference
            BuildableIdentifier = "primary"
            BlueprintIdentifier = "%s"
            BuildableName = "%s"
            BlueprintName = "%s"
            ReferencedContainer = "container:GoAI.xcodeproj">
         </BuildableReference>
      </BuildableProductRunnable>
   </ProfileAction>
   <AnalyzeAction buildConfiguration = "Debug"></AnalyzeAction>
   <ArchiveAction buildConfiguration = "Release" revealArchiveInOrganizer = "YES"></ArchiveAction>
</Scheme>
'''
for tgt, name in (('GoAI', 'GoAI'), ('Tests', 'GoAITests'), ('Board', 'GoBoard')):
    t = uid('target:' + tgt)
    open('GoAI.xcodeproj/xcshareddata/xcschemes/%s.xcscheme' % name, 'w').write(SCHEME % (t, name, name, t, name, name, t, name, name))
print('generated GoAI.xcodeproj')
