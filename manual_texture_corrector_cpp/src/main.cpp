#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <shellapi.h>
#include <propidl.h>
#include <gdiplus.h>

#include <algorithm>
#include <cmath>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <set>
#include <string>
#include <vector>

using namespace Gdiplus;
namespace fs = std::filesystem;

enum ControlId : int {
    IDC_PREV = 1001,
    IDC_NEXT = 1002,
    IDC_SAVE = 1003,
    IDC_SAVE_NEXT = 1004,
    IDC_OPEN_OUTPUT = 1005,
    IDC_OPEN_INPUT = 1006,
    IDC_ADD_VGUIDE = 1007,
    IDC_ADD_HGUIDE = 1008,
    IDC_CLEAR_GUIDES = 1009,
    IDC_REFRESH = 1010,
    IDC_FINISH = 1011,
    IDC_EDIT_BASE = 2001,
    IDC_COMBO_SUFFIX = 2002,
    IDC_EDIT_CUSTOM_SUFFIX = 2003,
    IDC_COMBO_EXT = 2004,
    IDC_EDIT_COLS = 2005,
    IDC_EDIT_ROWS = 2006,
    IDC_EDIT_EXPAND = 2007,
    IDC_CHECK_SAVE_ALL = 2008,
    IDC_EDIT_TILE_INDEX = 2009,
    IDC_EDIT_OUTPUT = 2010,
    IDC_EDIT_SNAP = 2011,
    IDC_COMBO_CHANNEL = 2012,
    IDC_EDIT_CUSTOM_CHANNEL = 2013,
    IDC_CHECK_SELECT_TILES = 2014,
    IDC_CHECK_SKIP_SOLID = 2015,
    IDC_COMBO_SELECTION_MODE = 2016,
    IDC_SPIN_COLS = 2101,
    IDC_SPIN_ROWS = 2102,
    IDC_SPIN_EXPAND = 2103,
    IDC_SPIN_TILE_INDEX = 2104,
    IDC_SPIN_SNAP = 2105,
    IDC_LABEL_FILE = 3001,
    IDC_LABEL_STATUS = 3002,
    IDC_LABEL_BASE = 3003,
    IDC_LABEL_SUFFIX = 3004,
    IDC_LABEL_EXT = 3005,
    IDC_LABEL_GRID = 3006,
    IDC_LABEL_SNAP = 3007,
    IDC_LABEL_TILE = 3008,
    IDC_LABEL_OUTPUT = 3009,
    IDC_LABEL_CHANNEL = 3010,
    IDC_LABEL_SELECTION_MODE = 3011,
};

enum class GuideMode { None, Vertical, Horizontal };
enum class SelectionExportMode { SplitSelected = 0, MergeSelected = 1 };

struct SuffixOption {
    const wchar_t* label;
    const wchar_t* token;
    const wchar_t* channel;
};

struct ChannelOption {
    const wchar_t* label;
    const wchar_t* token;
};

struct SelectionModeOption {
    const wchar_t* label;
    SelectionExportMode mode;
};

static constexpr SuffixOption kSuffixes[] = {
    {L"D (Diffuse)", L"D", L"diffuse"},
    {L"NM (Normal)", L"NM", L"normal"},
    {L"R (Roughness)", L"R", L"roughness"},
    {L"M (Metallic)", L"M", L"metallic"},
    {L"AO (Occlusion)", L"AO", L"occlusion"},
    {L"A (Alpha)", L"A", L"alpha"},
    {L"ORM", L"ORM", L"roughness"},
    {L"Custom", L"CUSTOM", L""},
};

static constexpr ChannelOption kChannels[] = {
    {L"Auto (from suffix)", L"AUTO"},
    {L"Diffuse", L"diffuse"},
    {L"Normal", L"normal"},
    {L"Roughness", L"roughness"},
    {L"Metallic", L"metallic"},
    {L"AO / Occlusion", L"occlusion"},
    {L"Alpha", L"alpha"},
    {L"Custom", L"CUSTOM"},
};

static constexpr SelectionModeOption kSelectionModes[] = {
    {L"Split selected tiles", SelectionExportMode::SplitSelected},
    {L"Merge selected tiles (one image)", SelectionExportMode::MergeSelected},
};

static constexpr const wchar_t* kExts[] = {L"png", L"jpg", L"bmp", L"tiff"};

struct ManifestEntry {
    std::wstring source;
    std::wstring base;
    std::wstring suffix;
    std::wstring channel;
    std::wstring extension;
    std::wstring primary;
    std::vector<std::wstring> outputs;
};

struct App {
    HWND hwnd = nullptr;
    HWND label_file = nullptr;
    HWND label_status = nullptr;
    HWND edit_base = nullptr;
    HWND combo_suffix = nullptr;
    HWND edit_custom_suffix = nullptr;
    HWND combo_channel = nullptr;
    HWND edit_custom_channel = nullptr;
    HWND combo_ext = nullptr;
    HWND edit_cols = nullptr;
    HWND edit_rows = nullptr;
    HWND edit_expand = nullptr;
    HWND edit_snap = nullptr;
    HWND check_save_all = nullptr;
    HWND check_select_tiles = nullptr;
    HWND check_skip_solid = nullptr;
    HWND combo_selection_mode = nullptr;
    HWND edit_tile_index = nullptr;
    HWND edit_output = nullptr;
    HWND spin_cols = nullptr;
    HWND spin_rows = nullptr;
    HWND spin_expand = nullptr;
    HWND spin_tile_index = nullptr;
    HWND spin_snap = nullptr;
    HWND label_base = nullptr;
    HWND label_suffix = nullptr;
    HWND label_channel = nullptr;
    HWND label_ext = nullptr;
    HWND label_grid = nullptr;
    HWND label_snap = nullptr;
    HWND label_tile = nullptr;
    HWND label_output = nullptr;
    HWND label_selection_mode = nullptr;

    std::wstring input_dir;
    std::wstring output_dir;
    std::vector<fs::path> files;
    int index = 0;
    std::unique_ptr<Bitmap> bmp;

    int cols = 2;
    int rows = 2;
    int expand_px = 0;
    int snap_px = 512;
    bool save_all = true;
    bool select_tiles_mode = false;
    bool skip_solid_tiles = true;
    SelectionExportMode selection_export_mode = SelectionExportMode::SplitSelected;
    int tile_index = 1;
    std::set<int> selected_tiles;
    std::vector<double> v_guides;
    std::vector<double> h_guides;
    GuideMode guide_mode = GuideMode::None;
    Rect canvas{0, 0, 0, 0};
    RectF draw_rect{0, 0, 0, 0};
    bool has_draw = false;

    std::vector<ManifestEntry> manifest;
};

static App g;

static std::wstring low(std::wstring s) {
    std::transform(s.begin(), s.end(), s.begin(), [](wchar_t ch) { return static_cast<wchar_t>(std::towlower(ch)); });
    return s;
}

static std::wstring trim(const std::wstring& s) {
    size_t a = 0;
    while (a < s.size() && std::iswspace(s[a])) {
        ++a;
    }
    size_t b = s.size();
    while (b > a && std::iswspace(s[b - 1])) {
        --b;
    }
    return s.substr(a, b - a);
}

static std::wstring get_text(HWND h) {
    if (!h) return L"";
    int len = GetWindowTextLengthW(h);
    if (len <= 0) return L"";
    std::vector<wchar_t> buf(static_cast<size_t>(len) + 1, L'\0');
    GetWindowTextW(h, buf.data(), len + 1);
    return std::wstring(buf.data());
}

static void set_text(HWND h, const std::wstring& v) { SetWindowTextW(h, v.c_str()); }

static int get_int(HWND h, int dflt) {
    try { return std::stoi(trim(get_text(h))); } catch (...) { return dflt; }
}

static void set_int(HWND h, int v) { set_text(h, std::to_wstring(v)); }

static int combo_idx(HWND h) { return static_cast<int>(SendMessageW(h, CB_GETCURSEL, 0, 0)); }

static int selection_mode_idx(SelectionExportMode mode) {
    for (size_t i = 0; i < std::size(kSelectionModes); ++i) {
        if (kSelectionModes[i].mode == mode) return static_cast<int>(i);
    }
    return 0;
}

static SelectionExportMode selection_mode_from_idx(int idx) {
    if (idx < 0 || idx >= static_cast<int>(std::size(kSelectionModes))) {
        return SelectionExportMode::SplitSelected;
    }
    return kSelectionModes[static_cast<size_t>(idx)].mode;
}

static void status(const std::wstring& msg) { if (g.label_status) set_text(g.label_status, msg); }

static bool is_img_ext(const fs::path& p) {
    const auto e = low(p.extension().wstring());
    return e == L".png" || e == L".jpg" || e == L".jpeg" || e == L".bmp" || e == L".tga" || e == L".tif" || e == L".tiff";
}

static bool ends_icase(const std::wstring& s, const std::wstring& suffix) {
    if (suffix.size() > s.size()) return false;
    return low(s.substr(s.size() - suffix.size())) == low(suffix);
}

static std::wstring strip_suffix(const std::wstring& stem) {
    std::wstring out = stem;
    const std::vector<std::wstring> toks = {
        L"_DIFFUSE", L"_ALBEDO", L"_COLOR", L"_NORMAL", L"_ROUGHNESS", L"_METALLIC", L"_OCCLUSION", L"_ALPHA",
        L"_ORM", L"_NM", L"_AO", L"_R", L"_M", L"_A", L"_D"
    };
    for (const auto& t : toks) {
        if (ends_icase(out, t)) {
            out = out.substr(0, out.size() - t.size());
            break;
        }
    }
    while (!out.empty() && out.back() == L'_') out.pop_back();
    return out.empty() ? stem : out;
}

static std::wstring guess_suffix(const std::wstring& stem) {
    if (ends_icase(stem, L"_NORMAL") || ends_icase(stem, L"_NM")) return L"NM";
    if (ends_icase(stem, L"_ROUGHNESS") || ends_icase(stem, L"_R")) return L"R";
    if (ends_icase(stem, L"_METALLIC") || ends_icase(stem, L"_M")) return L"M";
    if (ends_icase(stem, L"_OCCLUSION") || ends_icase(stem, L"_AO")) return L"AO";
    if (ends_icase(stem, L"_ALPHA") || ends_icase(stem, L"_A")) return L"A";
    if (ends_icase(stem, L"_ORM")) return L"ORM";
    return L"D";
}

static int suffix_idx_from_token(const std::wstring& token) {
    auto t = low(trim(token));
    for (size_t i = 0; i < std::size(kSuffixes); ++i) if (low(kSuffixes[i].token) == t) return static_cast<int>(i);
    return 0;
}

static int channel_idx_from_token(const std::wstring& token) {
    auto t = low(trim(token));
    if (t.empty()) return 0;
    for (size_t i = 0; i < std::size(kChannels); ++i) {
        if (low(kChannels[i].token) == t) return static_cast<int>(i);
    }
    return static_cast<int>(std::size(kChannels)) - 1;
}

static std::wstring suffix_token_from_ui() {
    std::wstring custom = trim(get_text(g.edit_custom_suffix));
    if (!custom.empty()) {
        return custom;
    }
    int i = combo_idx(g.combo_suffix);
    if (i < 0 || i >= static_cast<int>(std::size(kSuffixes))) return L"D";
    std::wstring tok = kSuffixes[i].token;
    return (tok == L"CUSTOM") ? L"D" : tok;
}

static std::wstring channel_from_suffix(const std::wstring& suffix) {
    auto t = low(trim(suffix));
    if (t == L"d" || t == L"diffuse" || t == L"color" || t == L"albedo") return L"diffuse";
    if (t == L"nm" || t == L"normal" || t == L"n") return L"normal";
    if (t == L"r" || t == L"roughness") return L"roughness";
    if (t == L"m" || t == L"metallic") return L"metallic";
    if (t == L"ao" || t == L"occlusion") return L"occlusion";
    if (t == L"a" || t == L"alpha") return L"alpha";
    if (t == L"orm") return L"roughness";
    return L"";
}

static std::wstring channel_from_ui(const std::wstring& suffix) {
    std::wstring custom = low(trim(get_text(g.edit_custom_channel)));
    if (!custom.empty()) {
        return custom;
    }
    int i = combo_idx(g.combo_channel);
    if (i < 0 || i >= static_cast<int>(std::size(kChannels))) {
        return channel_from_suffix(suffix);
    }
    std::wstring tok = kChannels[i].token;
    if (tok == L"AUTO") {
        return channel_from_suffix(suffix);
    }
    if (tok == L"CUSTOM") {
        return channel_from_suffix(suffix);
    }
    return tok;
}

static std::wstring ext_from_ui() {
    int i = combo_idx(g.combo_ext);
    if (i < 0 || i >= static_cast<int>(std::size(kExts))) return L"png";
    return kExts[i];
}

static std::string w2utf8(const std::wstring& ws) {
    if (ws.empty()) return {};
    int sz = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()), nullptr, 0, nullptr, nullptr);
    if (sz <= 0) return {};
    std::string out(static_cast<size_t>(sz), '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()), out.data(), sz, nullptr, nullptr);
    return out;
}

static std::string json_escape(const std::wstring& ws) {
    auto s = w2utf8(ws);
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << static_cast<char>(c); break;
        }
    }
    return oss.str();
}

static CLSID encoder_for_ext(const std::wstring& ext) {
    std::wstring mime = L"image/png";
    const auto e = low(ext);
    if (e == L"jpg" || e == L"jpeg") mime = L"image/jpeg";
    else if (e == L"bmp") mime = L"image/bmp";
    else if (e == L"tif" || e == L"tiff") mime = L"image/tiff";

    UINT num = 0, size = 0;
    GetImageEncodersSize(&num, &size);
    if (size == 0) return CLSID{};
    std::vector<BYTE> data(size);
    auto* codecs = reinterpret_cast<ImageCodecInfo*>(data.data());
    GetImageEncoders(num, size, codecs);
    for (UINT i = 0; i < num; ++i) {
        if (codecs[i].MimeType && _wcsicmp(codecs[i].MimeType, mime.c_str()) == 0) return codecs[i].Clsid;
    }
    return codecs[0].Clsid;
}

static bool save_crop(Bitmap& src, const Rect& sr, const fs::path& dst, const std::wstring& ext) {
    if (sr.Width <= 0 || sr.Height <= 0) return false;
    Bitmap tile(sr.Width, sr.Height, PixelFormat32bppARGB);
    Graphics gg(&tile);
    gg.SetInterpolationMode(InterpolationModeHighQualityBicubic);
    gg.DrawImage(&src, Rect(0, 0, sr.Width, sr.Height), sr.X, sr.Y, sr.Width, sr.Height, UnitPixel);
    fs::create_directories(dst.parent_path());
    CLSID clsid = encoder_for_ext(ext);
    return tile.Save(dst.wstring().c_str(), &clsid, nullptr) == Ok;
}

static void write_manifest(const fs::path& out_path) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"input_dir\": \"" << json_escape(g.input_dir) << "\",\n";
    ss << "  \"output_dir\": \"" << json_escape(g.output_dir) << "\",\n";
    ss << "  \"entries\": [\n";
    for (size_t i = 0; i < g.manifest.size(); ++i) {
        const auto& e = g.manifest[i];
        ss << "    {\n";
        ss << "      \"source\": \"" << json_escape(e.source) << "\",\n";
        ss << "      \"base\": \"" << json_escape(e.base) << "\",\n";
        ss << "      \"suffix\": \"" << json_escape(e.suffix) << "\",\n";
        ss << "      \"channel\": \"" << json_escape(e.channel) << "\",\n";
        ss << "      \"extension\": \"" << json_escape(e.extension) << "\",\n";
        ss << "      \"primary\": \"" << json_escape(e.primary) << "\",\n";
        ss << "      \"outputs\": [";
        for (size_t j = 0; j < e.outputs.size(); ++j) {
            if (j > 0) ss << ", ";
            ss << "\"" << json_escape(e.outputs[j]) << "\"";
        }
        ss << "]\n";
        ss << "    }" << ((i + 1 < g.manifest.size()) ? "," : "") << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";
    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    f << ss.str();
}

static void upsert_manifest(const ManifestEntry& e) {
    auto key = low(e.source);
    auto it = std::find_if(g.manifest.begin(), g.manifest.end(), [&](const ManifestEntry& cur) {
        return low(cur.source) == key;
    });
    if (it != g.manifest.end()) *it = e;
    else g.manifest.push_back(e);
}

static void sort_guides(std::vector<double>& guides) {
    std::sort(guides.begin(), guides.end());
    std::vector<double> out;
    for (double x : guides) {
        if (x <= 0.0 || x >= 1.0) continue;
        if (out.empty() || std::fabs(out.back() - x) > 1e-3) out.push_back(x);
    }
    guides.swap(out);
}

static std::vector<double> cuts_norm(int n, const std::vector<double>& guides) {
    std::vector<double> c;
    c.push_back(0.0);
    if (!guides.empty()) {
        for (double x : guides) if (x > 0.0 && x < 1.0) c.push_back(x);
        c.push_back(1.0);
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end(), [](double a, double b) { return std::fabs(a - b) < 1e-5; }), c.end());
        if (c.size() >= 2) return c;
    }
    c.clear();
    n = std::max(1, n);
    for (int i = 0; i <= n; ++i) c.push_back(static_cast<double>(i) / static_cast<double>(n));
    return c;
}

static std::vector<Rect> build_tiles(int w, int h) {
    auto xs = cuts_norm(g.cols, g.v_guides);
    auto ys = cuts_norm(g.rows, g.h_guides);
    std::vector<Rect> tiles;
    for (size_t yi = 0; yi + 1 < ys.size(); ++yi) {
        for (size_t xi = 0; xi + 1 < xs.size(); ++xi) {
            int x0 = static_cast<int>(std::floor(xs[xi] * w));
            int x1 = static_cast<int>(std::floor(xs[xi + 1] * w));
            int y0 = static_cast<int>(std::floor(ys[yi] * h));
            int y1 = static_cast<int>(std::floor(ys[yi + 1] * h));
            x0 = std::clamp(x0 - g.expand_px, 0, w);
            y0 = std::clamp(y0 - g.expand_px, 0, h);
            x1 = std::clamp(x1 + g.expand_px, 0, w);
            y1 = std::clamp(y1 + g.expand_px, 0, h);
            if (x1 > x0 && y1 > y0) tiles.emplace_back(x0, y0, x1 - x0, y1 - y0);
        }
    }
    return tiles;
}

static void prune_selected_tiles(size_t total) {
    for (auto it = g.selected_tiles.begin(); it != g.selected_tiles.end();) {
        if (*it < 0 || static_cast<size_t>(*it) >= total) {
            it = g.selected_tiles.erase(it);
        } else {
            ++it;
        }
    }
}

static Rect selected_bbox(const std::vector<Rect>& tiles, const std::set<int>& selected) {
    bool has = false;
    int min_x = 0, min_y = 0, max_x = 0, max_y = 0;
    for (int idx : selected) {
        if (idx < 0 || static_cast<size_t>(idx) >= tiles.size()) continue;
        const Rect& r = tiles[static_cast<size_t>(idx)];
        if (!has) {
            min_x = r.X;
            min_y = r.Y;
            max_x = r.X + r.Width;
            max_y = r.Y + r.Height;
            has = true;
            continue;
        }
        min_x = std::min(min_x, r.X);
        min_y = std::min(min_y, r.Y);
        max_x = std::max(max_x, r.X + r.Width);
        max_y = std::max(max_y, r.Y + r.Height);
    }
    if (!has) return Rect(0, 0, 0, 0);
    return Rect(min_x, min_y, std::max(0, max_x - min_x), std::max(0, max_y - min_y));
}

static bool is_uniform_color_tile(Bitmap& src, const Rect& sr, int tolerance = 2) {
    if (sr.Width <= 0 || sr.Height <= 0) return true;
    std::unique_ptr<Bitmap> tile(src.Clone(sr, PixelFormat32bppARGB));
    if (!tile || tile->GetLastStatus() != Ok) return false;

    Rect tr(0, 0, static_cast<INT>(tile->GetWidth()), static_cast<INT>(tile->GetHeight()));
    BitmapData data{};
    if (tile->LockBits(&tr, ImageLockModeRead, PixelFormat32bppARGB, &data) != Ok) {
        return false;
    }

    const int w = tr.Width;
    const int h = tr.Height;
    const int stride = data.Stride;
    const auto* base = static_cast<const BYTE*>(data.Scan0);
    int minv[4] = {255, 255, 255, 255};
    int maxv[4] = {0, 0, 0, 0};
    bool uniform = true;

    for (int y = 0; y < h && uniform; ++y) {
        const BYTE* row = base + y * stride;
        for (int x = 0; x < w; ++x) {
            const BYTE* px = row + x * 4;
            for (int c = 0; c < 4; ++c) {
                int v = static_cast<int>(px[c]);
                minv[c] = std::min(minv[c], v);
                maxv[c] = std::max(maxv[c], v);
                if (maxv[c] - minv[c] > tolerance) {
                    uniform = false;
                    break;
                }
            }
            if (!uniform) break;
        }
    }

    tile->UnlockBits(&data);
    return uniform;
}

static void refresh_file_label() {
    if (g.files.empty()) {
        set_text(g.label_file, L"No images found.");
        return;
    }
    std::wstringstream ss;
    ss << (g.index + 1) << L"/" << g.files.size() << L"  " << g.files[g.index].filename().wstring();
    set_text(g.label_file, ss.str());
}

static void ui_to_state() {
    g.cols = std::max(1, get_int(g.edit_cols, g.cols));
    g.rows = std::max(1, get_int(g.edit_rows, g.rows));
    g.expand_px = std::max(0, get_int(g.edit_expand, g.expand_px));
    g.snap_px = std::max(1, get_int(g.edit_snap, g.snap_px));
    g.save_all = SendMessageW(g.check_save_all, BM_GETCHECK, 0, 0) == BST_CHECKED;
    g.select_tiles_mode = SendMessageW(g.check_select_tiles, BM_GETCHECK, 0, 0) == BST_CHECKED;
    g.skip_solid_tiles = SendMessageW(g.check_skip_solid, BM_GETCHECK, 0, 0) == BST_CHECKED;
    g.selection_export_mode = selection_mode_from_idx(combo_idx(g.combo_selection_mode));
    g.tile_index = std::max(1, get_int(g.edit_tile_index, g.tile_index));
    g.output_dir = trim(get_text(g.edit_output));
}

static void state_to_ui() {
    set_int(g.edit_cols, g.cols);
    set_int(g.edit_rows, g.rows);
    set_int(g.edit_expand, g.expand_px);
    set_int(g.edit_snap, g.snap_px);
    SendMessageW(g.check_save_all, BM_SETCHECK, g.save_all ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessageW(g.check_select_tiles, BM_SETCHECK, g.select_tiles_mode ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessageW(g.check_skip_solid, BM_SETCHECK, g.skip_solid_tiles ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessageW(g.combo_selection_mode, CB_SETCURSEL, selection_mode_idx(g.selection_export_mode), 0);
    set_int(g.edit_tile_index, g.tile_index);
    set_text(g.edit_output, g.output_dir);
}

static void update_custom_suffix_enabled() {
    if (!g.edit_custom_suffix) {
        return;
    }
    EnableWindow(g.edit_custom_suffix, TRUE);
}

static void update_custom_channel_enabled() {
    if (!g.edit_custom_channel) {
        return;
    }
    EnableWindow(g.edit_custom_channel, TRUE);
}

static void sync_suffix_editor_from_combo() {
    if (!g.combo_suffix || !g.edit_custom_suffix) return;
    int i = combo_idx(g.combo_suffix);
    if (i < 0 || i >= static_cast<int>(std::size(kSuffixes))) return;
    const std::wstring tok = kSuffixes[i].token;
    if (low(tok) == L"custom") return;
    set_text(g.edit_custom_suffix, tok);
}

static void sync_channel_editor_from_combo() {
    if (!g.combo_channel || !g.edit_custom_channel) return;
    int i = combo_idx(g.combo_channel);
    if (i < 0 || i >= static_cast<int>(std::size(kChannels))) return;
    const std::wstring tok = kChannels[i].token;
    if (low(tok) == L"custom") return;
    if (low(tok) == L"auto") {
        set_text(g.edit_custom_channel, channel_from_suffix(suffix_token_from_ui()));
        return;
    }
    set_text(g.edit_custom_channel, tok);
}

static void update_tile_index_enabled() {
    const BOOL on = (!g.save_all && !g.select_tiles_mode) ? TRUE : FALSE;
    if (g.check_save_all) {
        EnableWindow(g.check_save_all, g.select_tiles_mode ? FALSE : TRUE);
    }
    if (g.combo_selection_mode) {
        EnableWindow(g.combo_selection_mode, g.select_tiles_mode ? TRUE : FALSE);
    }
    if (g.label_selection_mode) {
        EnableWindow(g.label_selection_mode, g.select_tiles_mode ? TRUE : FALSE);
    }
    if (g.edit_tile_index) {
        EnableWindow(g.edit_tile_index, on);
    }
    if (g.spin_tile_index) {
        EnableWindow(g.spin_tile_index, on);
    }
    if (g.label_tile) {
        EnableWindow(g.label_tile, on);
    }
}

static void defaults_from_name() {
    if (g.files.empty()) return;
    auto stem = g.files[g.index].stem().wstring();
    auto suff = guess_suffix(stem);
    auto base = strip_suffix(stem);
    set_text(g.edit_base, base);
    SendMessageW(g.combo_suffix, CB_SETCURSEL, suffix_idx_from_token(suff), 0);
    set_text(g.edit_custom_suffix, suff);
    update_custom_suffix_enabled();
    SendMessageW(g.combo_channel, CB_SETCURSEL, channel_idx_from_token(L"AUTO"), 0);
    set_text(g.edit_custom_channel, channel_from_suffix(suff));
    update_custom_channel_enabled();
}

static bool load_current() {
    g.bmp.reset();
    g.v_guides.clear();
    g.h_guides.clear();
    g.selected_tiles.clear();
    g.guide_mode = GuideMode::None;
    if (g.files.empty()) {
        refresh_file_label();
        InvalidateRect(g.hwnd, nullptr, TRUE);
        return false;
    }
    g.index = std::clamp(g.index, 0, static_cast<int>(g.files.size()) - 1);
    auto bmp = std::make_unique<Bitmap>(g.files[g.index].wstring().c_str());
    if (bmp->GetLastStatus() != Ok || bmp->GetWidth() == 0 || bmp->GetHeight() == 0) {
        status(L"Failed to load image.");
        refresh_file_label();
        InvalidateRect(g.hwnd, nullptr, TRUE);
        return false;
    }
    g.bmp = std::move(bmp);
    defaults_from_name();
    refresh_file_label();
    status(L"Ready.");
    InvalidateRect(g.hwnd, nullptr, TRUE);
    return true;
}

static void scan_images() {
    g.files.clear();
    g.index = 0;
    try {
        for (const auto& e : fs::directory_iterator(fs::path(g.input_dir))) {
            if (e.is_regular_file() && is_img_ext(e.path())) g.files.push_back(e.path());
        }
        std::sort(g.files.begin(), g.files.end(), [](const fs::path& a, const fs::path& b) {
            return low(a.filename().wstring()) < low(b.filename().wstring());
        });
    } catch (...) {
        g.files.clear();
    }
    load_current();
}

static void open_folder(const std::wstring& folder) {
    if (!folder.empty()) ShellExecuteW(nullptr, L"open", folder.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

static void save_current(bool and_next) {
    ui_to_state();
    if (!g.bmp || g.files.empty()) { status(L"No image loaded."); return; }
    std::wstring base = trim(get_text(g.edit_base));
    if (base.empty()) {
        base = strip_suffix(g.files[g.index].stem().wstring());
        set_text(g.edit_base, base);
    }
    std::wstring suffix = suffix_token_from_ui();
    if (suffix.empty()) suffix = L"D";
    std::wstring ext = low(ext_from_ui());
    if (ext.empty()) ext = L"png";

    fs::path out = g.output_dir.empty() ? fs::path(g.input_dir) / L"manual_corrected" : fs::path(g.output_dir);
    g.output_dir = out.wstring();
    set_text(g.edit_output, g.output_dir);
    try { fs::create_directories(out); } catch (...) { status(L"Cannot create output folder."); return; }

    int w = static_cast<int>(g.bmp->GetWidth());
    int h = static_cast<int>(g.bmp->GetHeight());
    auto tiles = build_tiles(w, h);
    if (tiles.empty()) { status(L"No valid tiles."); return; }
    prune_selected_tiles(tiles.size());

    std::vector<int> pick;
    if (g.select_tiles_mode) {
        if (g.selected_tiles.empty()) {
            status(L"No tiles selected. Enable tile selection and click tiles on image.");
            return;
        }
        for (int idx : g.selected_tiles) pick.push_back(idx);
    } else if (g.save_all) {
        pick.resize(tiles.size());
        for (size_t i = 0; i < tiles.size(); ++i) pick[i] = static_cast<int>(i);
    } else {
        pick.push_back(std::clamp(g.tile_index - 1, 0, static_cast<int>(tiles.size()) - 1));
    }

    std::vector<std::wstring> outputs;
    int skipped_solid = 0;
    const bool use_merge_mode = g.select_tiles_mode && g.selection_export_mode == SelectionExportMode::MergeSelected;
    if (use_merge_mode) {
        Rect bbox = selected_bbox(tiles, g.selected_tiles);
        if (bbox.Width <= 0 || bbox.Height <= 0) {
            status(L"Save failed: selected area is empty.");
            return;
        }
        if (g.skip_solid_tiles && is_uniform_color_tile(*g.bmp, bbox)) {
            status(L"Nothing saved: merged selected area is solid-color.");
            return;
        }
        std::wstring stem = base + L"_" + suffix + L"_MERGED";
        fs::path dst = out / (stem + L"." + ext);
        if (save_crop(*g.bmp, bbox, dst, ext)) {
            outputs.push_back(dst.wstring());
        }
    } else {
        size_t saved_idx = 0;
        for (size_t i = 0; i < pick.size(); ++i) {
            if (g.skip_solid_tiles && is_uniform_color_tile(*g.bmp, tiles[pick[i]])) {
                ++skipped_solid;
                continue;
            }
            std::wstring stem = base + L"_" + suffix;
            if (pick.size() > 1) stem += L"_part" + std::to_wstring(saved_idx + 1);
            fs::path dst = out / (stem + L"." + ext);
            if (save_crop(*g.bmp, tiles[pick[i]], dst, ext)) {
                outputs.push_back(dst.wstring());
                ++saved_idx;
            }
        }
    }
    if (outputs.empty()) {
        if (skipped_solid > 0) status(L"Nothing saved: all selected tiles are solid-color.");
        else status(L"Save failed.");
        return;
    }

    ManifestEntry e;
    e.source = g.files[g.index].wstring();
    e.base = base;
    e.suffix = suffix;
    e.channel = channel_from_ui(suffix);
    e.extension = ext;
    e.outputs = outputs;
    e.primary = outputs.front();
    upsert_manifest(e);
    write_manifest(out / L"manual_texture_manifest.json");

    std::wstringstream ss;
    const wchar_t* mode_name = use_merge_mode ? L"Merge" : L"Split";
    ss << mode_name << L": saved " << outputs.size() << L" tile(s)";
    if (skipped_solid > 0) ss << L", skipped solid: " << skipped_solid;
    ss << L": " << fs::path(e.primary).filename().wstring();
    status(ss.str());
    if (and_next && g.index + 1 < static_cast<int>(g.files.size())) { ++g.index; load_current(); }
}

static void layout(HWND hwnd) {
    RECT rc{};
    GetClientRect(hwnd, &rc);
    int w = rc.right - rc.left;
    int h = rc.bottom - rc.top;
    int panel_w = 420;
    int px = std::max(0, w - panel_w) + 10;
    int pw = panel_w - 20;
    g.canvas = Rect(10, 10, std::max(120, w - panel_w - 20), std::max(120, h - 20));

    int y = 10;
    auto place = [&](HWND c, int hh = 24) { MoveWindow(c, px, y, pw, hh, TRUE); y += hh + 6; };
    auto place_combo = [&](HWND c, int visible_h = 24, int drop_h = 220) {
        MoveWindow(c, px, y, pw, drop_h, TRUE);
        y += visible_h + 6;
    };
    auto place_label = [&](HWND c) { MoveWindow(c, px, y, pw, 18, TRUE); y += 20; };
    auto place2 = [&](HWND a, HWND b, int hh = 24) {
        int gap = 6, w2 = (pw - gap) / 2;
        MoveWindow(a, px, y, w2, hh, TRUE);
        MoveWindow(b, px + w2 + gap, y, pw - w2 - gap, hh, TRUE);
        y += hh + 6;
    };
    auto place_combo2 = [&](HWND combo, HWND edit, int visible_h = 24, int drop_h = 220) {
        int gap = 6, w2 = (pw - gap) / 2;
        MoveWindow(combo, px, y, w2, drop_h, TRUE);
        MoveWindow(edit, px + w2 + gap, y, pw - w2 - gap, visible_h, TRUE);
        y += visible_h + 6;
    };
    auto place3 = [&](HWND a, HWND b, HWND c, int hh = 24) {
        int gap = 6, w3 = (pw - 2 * gap) / 3;
        MoveWindow(a, px, y, w3, hh, TRUE);
        MoveWindow(b, px + w3 + gap, y, w3, hh, TRUE);
        MoveWindow(c, px + 2 * (w3 + gap), y, pw - 2 * (w3 + gap), hh, TRUE);
        y += hh + 6;
    };
    auto place_num3 = [&](HWND e1, HWND s1, HWND e2, HWND s2, HWND e3, HWND s3, int hh = 24) {
        int gap = 6;
        int w3 = (pw - 2 * gap) / 3;
        int spin_w = 22;
        int edit_w = std::max(28, w3 - spin_w);
        MoveWindow(e1, px, y, edit_w, hh, TRUE);
        MoveWindow(s1, px + edit_w, y, spin_w, hh, TRUE);
        MoveWindow(e2, px + w3 + gap, y, edit_w, hh, TRUE);
        MoveWindow(s2, px + w3 + gap + edit_w, y, spin_w, hh, TRUE);
        MoveWindow(e3, px + 2 * (w3 + gap), y, edit_w, hh, TRUE);
        MoveWindow(s3, px + 2 * (w3 + gap) + edit_w, y, spin_w, hh, TRUE);
        y += hh + 6;
    };
    auto place_num1 = [&](HWND e, HWND s, int hh = 24) {
        int spin_w = 22;
        MoveWindow(e, px, y, std::max(40, pw - spin_w), hh, TRUE);
        MoveWindow(s, px + std::max(40, pw - spin_w), y, spin_w, hh, TRUE);
        y += hh + 6;
    };

    place(g.label_file);
    place_label(g.label_base);
    place(g.edit_base);
    place_label(g.label_suffix);
    place_combo2(g.combo_suffix, g.edit_custom_suffix);
    place_label(g.label_channel);
    place_combo2(g.combo_channel, g.edit_custom_channel);
    place_label(g.label_ext);
    place_combo(g.combo_ext);
    place_label(g.label_grid);
    place_num3(g.edit_cols, g.spin_cols, g.edit_rows, g.spin_rows, g.edit_expand, g.spin_expand);
    place_label(g.label_snap);
    place_num1(g.edit_snap, g.spin_snap);
    place(g.check_save_all);
    place(g.check_select_tiles);
    place_label(g.label_selection_mode);
    place_combo(g.combo_selection_mode);
    place(g.check_skip_solid);
    place_label(g.label_tile);
    place_num1(g.edit_tile_index, g.spin_tile_index);
    place_label(g.label_output);
    place(g.edit_output);
    place2(GetDlgItem(hwnd, IDC_OPEN_INPUT), GetDlgItem(hwnd, IDC_OPEN_OUTPUT), 28);
    place3(GetDlgItem(hwnd, IDC_ADD_VGUIDE), GetDlgItem(hwnd, IDC_ADD_HGUIDE), GetDlgItem(hwnd, IDC_CLEAR_GUIDES), 28);
    place3(GetDlgItem(hwnd, IDC_PREV), GetDlgItem(hwnd, IDC_NEXT), GetDlgItem(hwnd, IDC_REFRESH), 28);
    place2(GetDlgItem(hwnd, IDC_SAVE), GetDlgItem(hwnd, IDC_SAVE_NEXT), 30);
    place(g.label_status, 56);
    place(GetDlgItem(hwnd, IDC_FINISH), 30);
}

static void draw(HDC hdc) {
    Graphics gr(hdc);
    gr.SetSmoothingMode(SmoothingModeAntiAlias);
    SolidBrush bg(Color(255, 26, 28, 32));
    gr.FillRectangle(&bg, g.canvas);
    Pen border(Color(255, 68, 72, 80), 1.0f);
    gr.DrawRectangle(&border, g.canvas);

    if (!g.bmp) {
        g.has_draw = false;
        return;
    }

    float sx = static_cast<float>(g.canvas.Width) / std::max(1.0f, static_cast<float>(g.bmp->GetWidth()));
    float sy = static_cast<float>(g.canvas.Height) / std::max(1.0f, static_cast<float>(g.bmp->GetHeight()));
    float s = std::min(sx, sy);
    float dw = static_cast<float>(g.bmp->GetWidth()) * s;
    float dh = static_cast<float>(g.bmp->GetHeight()) * s;
    float dx = static_cast<float>(g.canvas.X) + (static_cast<float>(g.canvas.Width) - dw) * 0.5f;
    float dy = static_cast<float>(g.canvas.Y) + (static_cast<float>(g.canvas.Height) - dh) * 0.5f;
    g.draw_rect = RectF(dx, dy, dw, dh);
    g.has_draw = true;
    gr.DrawImage(g.bmp.get(), g.draw_rect);

    auto xs = cuts_norm(g.cols, g.v_guides);
    auto ys = cuts_norm(g.rows, g.h_guides);
    const int tile_cols = static_cast<int>(xs.size()) - 1;
    const int tile_rows = static_cast<int>(ys.size()) - 1;
    const size_t tile_count = (tile_cols > 0 && tile_rows > 0) ? static_cast<size_t>(tile_cols * tile_rows) : 0;
    prune_selected_tiles(tile_count);

    if (!g.selected_tiles.empty()) {
        const bool use_merge_mode = g.select_tiles_mode && g.selection_export_mode == SelectionExportMode::MergeSelected;
        SolidBrush sel_fill(use_merge_mode ? Color(50, 220, 40, 40) : Color(70, 80, 220, 120));
        Pen sel_border(use_merge_mode ? Color(240, 255, 50, 50) : Color(220, 80, 220, 120), 2.0f);
        for (int idx : g.selected_tiles) {
            if (idx < 0 || idx >= tile_cols * tile_rows) continue;
            const int yi = idx / tile_cols;
            const int xi = idx % tile_cols;
            if (xi < 0 || yi < 0 || xi + 1 >= static_cast<int>(xs.size()) || yi + 1 >= static_cast<int>(ys.size())) continue;
            const float x0 = dx + static_cast<float>(xs[static_cast<size_t>(xi)]) * dw;
            const float x1 = dx + static_cast<float>(xs[static_cast<size_t>(xi + 1)]) * dw;
            const float y0 = dy + static_cast<float>(ys[static_cast<size_t>(yi)]) * dh;
            const float y1 = dy + static_cast<float>(ys[static_cast<size_t>(yi + 1)]) * dh;
            RectF rr(x0, y0, std::max(1.0f, x1 - x0), std::max(1.0f, y1 - y0));
            gr.FillRectangle(&sel_fill, rr);
            gr.DrawRectangle(&sel_border, rr);
        }
        if (use_merge_mode) {
            Rect bbox_px = selected_bbox(build_tiles(static_cast<int>(g.bmp->GetWidth()), static_cast<int>(g.bmp->GetHeight())), g.selected_tiles);
            if (bbox_px.Width > 0 && bbox_px.Height > 0) {
                const float bx0 = dx + (static_cast<float>(bbox_px.X) / std::max(1.0f, static_cast<float>(g.bmp->GetWidth()))) * dw;
                const float by0 = dy + (static_cast<float>(bbox_px.Y) / std::max(1.0f, static_cast<float>(g.bmp->GetHeight()))) * dh;
                const float bx1 = dx + (static_cast<float>(bbox_px.X + bbox_px.Width) / std::max(1.0f, static_cast<float>(g.bmp->GetWidth()))) * dw;
                const float by1 = dy + (static_cast<float>(bbox_px.Y + bbox_px.Height) / std::max(1.0f, static_cast<float>(g.bmp->GetHeight()))) * dh;
                Pen bbox_pen(Color(255, 255, 0, 0), 3.0f);
                gr.DrawRectangle(&bbox_pen, RectF(bx0, by0, std::max(1.0f, bx1 - bx0), std::max(1.0f, by1 - by0)));
            }
        }
    }

    Pen p(Color(180, 255, 155, 70), 1.5f);
    for (size_t i = 1; i + 1 < xs.size(); ++i) {
        float x = dx + static_cast<float>(xs[i]) * dw;
        gr.DrawLine(&p, x, dy, x, dy + dh);
    }
    for (size_t i = 1; i + 1 < ys.size(); ++i) {
        float y = dy + static_cast<float>(ys[i]) * dh;
        gr.DrawLine(&p, dx, y, dx + dw, y);
    }
}

static int tile_index_from_click(int mx, int my) {
    if (!g.has_draw || !g.bmp) return -1;
    auto r = g.draw_rect;
    if (mx < r.X || mx > r.X + r.Width || my < r.Y || my > r.Y + r.Height) return -1;

    const auto xs = cuts_norm(g.cols, g.v_guides);
    const auto ys = cuts_norm(g.rows, g.h_guides);
    if (xs.size() < 2 || ys.size() < 2) return -1;

    const double nx = (static_cast<double>(mx) - r.X) / std::max(1.0, static_cast<double>(r.Width));
    const double ny = (static_cast<double>(my) - r.Y) / std::max(1.0, static_cast<double>(r.Height));

    int xi = -1;
    int yi = -1;
    for (size_t i = 0; i + 1 < xs.size(); ++i) {
        if ((nx >= xs[i] && nx < xs[i + 1]) || (i + 2 == xs.size() && nx <= xs[i + 1])) {
            xi = static_cast<int>(i);
            break;
        }
    }
    for (size_t i = 0; i + 1 < ys.size(); ++i) {
        if ((ny >= ys[i] && ny < ys[i + 1]) || (i + 2 == ys.size() && ny <= ys[i + 1])) {
            yi = static_cast<int>(i);
            break;
        }
    }
    if (xi < 0 || yi < 0) return -1;
    const int cols = static_cast<int>(xs.size()) - 1;
    return yi * cols + xi;
}

static bool toggle_tile_click(int mx, int my) {
    if (!g.select_tiles_mode || !g.bmp) return false;
    const int idx = tile_index_from_click(mx, my);
    if (idx < 0) return false;
    auto it = g.selected_tiles.find(idx);
    if (it == g.selected_tiles.end()) {
        g.selected_tiles.insert(idx);
    } else {
        g.selected_tiles.erase(it);
    }
    std::wstringstream ss;
    ss << L"Selected tiles: " << g.selected_tiles.size();
    status(ss.str());
    InvalidateRect(g.hwnd, nullptr, FALSE);
    return true;
}

static bool add_guide_click(int mx, int my) {
    if (!g.has_draw || g.guide_mode == GuideMode::None) return false;
    auto r = g.draw_rect;
    if (mx < r.X || mx > r.X + r.Width || my < r.Y || my > r.Y + r.Height) return false;
    const int w = std::max(1, static_cast<int>(g.bmp ? g.bmp->GetWidth() : 1));
    const int h = std::max(1, static_cast<int>(g.bmp ? g.bmp->GetHeight() : 1));
    const int snap = std::max(1, g.snap_px);
    if (g.guide_mode == GuideMode::Vertical) {
        double px = ((static_cast<double>(mx) - r.X) / std::max(1.0, static_cast<double>(r.Width))) * static_cast<double>(w);
        if (snap > 1) {
            px = std::round(px / static_cast<double>(snap)) * static_cast<double>(snap);
        }
        px = std::clamp(px, 0.0, static_cast<double>(w));
        g.v_guides.push_back(px / static_cast<double>(w));
        sort_guides(g.v_guides);
        status(L"Vertical guide added at x=" + std::to_wstring(static_cast<int>(std::round(px))) + L" px");
    } else {
        double py = ((static_cast<double>(my) - r.Y) / std::max(1.0, static_cast<double>(r.Height))) * static_cast<double>(h);
        if (snap > 1) {
            py = std::round(py / static_cast<double>(snap)) * static_cast<double>(snap);
        }
        py = std::clamp(py, 0.0, static_cast<double>(h));
        g.h_guides.push_back(py / static_cast<double>(h));
        sort_guides(g.h_guides);
        status(L"Horizontal guide added at y=" + std::to_wstring(static_cast<int>(std::round(py))) + L" px");
    }
    g.guide_mode = GuideMode::None;
    InvalidateRect(g.hwnd, nullptr, FALSE);
    return true;
}

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        g.hwnd = hwnd;
        HFONT f = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
        auto mk_static = [&](int id, const wchar_t* t) { HWND h = CreateWindowExW(0, L"STATIC", t, WS_CHILD | WS_VISIBLE, 0, 0, 10, 10, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr); SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE); return h; };
        auto mk_edit = [&](int id, const wchar_t* t, DWORD extra_style = 0) { HWND h = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", t, WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | extra_style, 0, 0, 10, 10, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr); SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE); return h; };
        auto mk_btn = [&](int id, const wchar_t* t, DWORD st = BS_PUSHBUTTON) { HWND h = CreateWindowExW(0, L"BUTTON", t, WS_CHILD | WS_VISIBLE | st, 0, 0, 10, 10, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr); SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE); return h; };
        auto mk_spin = [&](int id, HWND buddy, int min_v, int max_v, int pos_v) {
            HWND h = CreateWindowExW(0, UPDOWN_CLASSW, L"", WS_CHILD | WS_VISIBLE | UDS_ARROWKEYS | UDS_SETBUDDYINT | UDS_NOTHOUSANDS, 0, 0, 10, 10, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr);
            SendMessageW(h, UDM_SETBUDDY, reinterpret_cast<WPARAM>(buddy), 0);
            SendMessageW(h, UDM_SETRANGE32, min_v, max_v);
            SendMessageW(h, UDM_SETPOS32, 0, pos_v);
            return h;
        };

        g.label_file = mk_static(IDC_LABEL_FILE, L"No files");
        g.label_base = mk_static(IDC_LABEL_BASE, L"Base name");
        g.edit_base = mk_edit(IDC_EDIT_BASE, L"");
        g.label_suffix = mk_static(IDC_LABEL_SUFFIX, L"Suffix");
        g.combo_suffix = CreateWindowExW(0, L"COMBOBOX", L"", WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST | CBS_NOINTEGRALHEIGHT, 0, 0, 10, 10, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_COMBO_SUFFIX)), nullptr, nullptr);
        SendMessageW(g.combo_suffix, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE);
        for (const auto& opt : kSuffixes) SendMessageW(g.combo_suffix, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(opt.label));
        SendMessageW(g.combo_suffix, CB_SETCURSEL, 0, 0);
        SendMessageW(g.combo_suffix, CB_SETMINVISIBLE, 8, 0);
        g.edit_custom_suffix = mk_edit(IDC_EDIT_CUSTOM_SUFFIX, L"");
        g.label_channel = mk_static(IDC_LABEL_CHANNEL, L"Channel");
        g.combo_channel = CreateWindowExW(0, L"COMBOBOX", L"", WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST | CBS_NOINTEGRALHEIGHT, 0, 0, 10, 10, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_COMBO_CHANNEL)), nullptr, nullptr);
        SendMessageW(g.combo_channel, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE);
        for (const auto& opt : kChannels) SendMessageW(g.combo_channel, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(opt.label));
        SendMessageW(g.combo_channel, CB_SETCURSEL, 0, 0);
        SendMessageW(g.combo_channel, CB_SETMINVISIBLE, 8, 0);
        g.edit_custom_channel = mk_edit(IDC_EDIT_CUSTOM_CHANNEL, L"");
        g.label_ext = mk_static(IDC_LABEL_EXT, L"Output extension");
        g.combo_ext = CreateWindowExW(0, L"COMBOBOX", L"", WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST | CBS_NOINTEGRALHEIGHT, 0, 0, 10, 10, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_COMBO_EXT)), nullptr, nullptr);
        SendMessageW(g.combo_ext, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE);
        for (auto ext : kExts) SendMessageW(g.combo_ext, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(ext));
        SendMessageW(g.combo_ext, CB_SETCURSEL, 0, 0);
        SendMessageW(g.combo_ext, CB_SETMINVISIBLE, 8, 0);
        g.label_grid = mk_static(IDC_LABEL_GRID, L"Grid settings: Columns | Rows | Expand (px)");
        g.edit_cols = mk_edit(IDC_EDIT_COLS, L"2", ES_NUMBER);
        g.spin_cols = mk_spin(IDC_SPIN_COLS, g.edit_cols, 1, 64, 2);
        g.edit_rows = mk_edit(IDC_EDIT_ROWS, L"2", ES_NUMBER);
        g.spin_rows = mk_spin(IDC_SPIN_ROWS, g.edit_rows, 1, 64, 2);
        g.edit_expand = mk_edit(IDC_EDIT_EXPAND, L"0", ES_NUMBER);
        g.spin_expand = mk_spin(IDC_SPIN_EXPAND, g.edit_expand, 0, 8192, 0);
        g.label_snap = mk_static(IDC_LABEL_SNAP, L"Guide snap step (px), e.g. 512/1024");
        g.edit_snap = mk_edit(IDC_EDIT_SNAP, L"512", ES_NUMBER);
        g.spin_snap = mk_spin(IDC_SPIN_SNAP, g.edit_snap, 1, 8192, 512);
        g.check_save_all = mk_btn(IDC_CHECK_SAVE_ALL, L"Save all tiles", BS_AUTOCHECKBOX);
        SendMessageW(g.check_save_all, BM_SETCHECK, BST_CHECKED, 0);
        g.check_select_tiles = mk_btn(IDC_CHECK_SELECT_TILES, L"Select tiles manually (click on image)", BS_AUTOCHECKBOX);
        SendMessageW(g.check_select_tiles, BM_SETCHECK, BST_UNCHECKED, 0);
        g.label_selection_mode = mk_static(IDC_LABEL_SELECTION_MODE, L"Selection export mode");
        g.combo_selection_mode = CreateWindowExW(0, L"COMBOBOX", L"", WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST | CBS_NOINTEGRALHEIGHT, 0, 0, 10, 10, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_COMBO_SELECTION_MODE)), nullptr, nullptr);
        SendMessageW(g.combo_selection_mode, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE);
        for (const auto& opt : kSelectionModes) SendMessageW(g.combo_selection_mode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(opt.label));
        SendMessageW(g.combo_selection_mode, CB_SETCURSEL, 0, 0);
        SendMessageW(g.combo_selection_mode, CB_SETMINVISIBLE, 2, 0);
        g.check_skip_solid = mk_btn(IDC_CHECK_SKIP_SOLID, L"Skip solid-color tiles", BS_AUTOCHECKBOX);
        SendMessageW(g.check_skip_solid, BM_SETCHECK, BST_CHECKED, 0);
        g.label_tile = mk_static(IDC_LABEL_TILE, L"Tile index (used when 'Save all tiles' is OFF)");
        g.edit_tile_index = mk_edit(IDC_EDIT_TILE_INDEX, L"1", ES_NUMBER);
        g.spin_tile_index = mk_spin(IDC_SPIN_TILE_INDEX, g.edit_tile_index, 1, 256, 1);
        g.label_output = mk_static(IDC_LABEL_OUTPUT, L"Output folder");
        g.edit_output = mk_edit(IDC_EDIT_OUTPUT, g.output_dir.c_str());
        mk_btn(IDC_OPEN_INPUT, L"Open input");
        mk_btn(IDC_OPEN_OUTPUT, L"Open output");
        mk_btn(IDC_ADD_VGUIDE, L"Add V guide");
        mk_btn(IDC_ADD_HGUIDE, L"Add H guide");
        mk_btn(IDC_CLEAR_GUIDES, L"Clear guides");
        mk_btn(IDC_PREV, L"Prev");
        mk_btn(IDC_NEXT, L"Next");
        mk_btn(IDC_REFRESH, L"Refresh");
        mk_btn(IDC_SAVE, L"Save current");
        mk_btn(IDC_SAVE_NEXT, L"Save + Next");
        g.label_status = mk_static(IDC_LABEL_STATUS, L"Ready.");
        mk_btn(IDC_FINISH, L"Finish / Close");
        state_to_ui();
        update_custom_suffix_enabled();
        update_custom_channel_enabled();
        update_tile_index_enabled();
        scan_images();
        layout(hwnd);
        return 0;
    }
    case WM_SIZE: layout(hwnd); InvalidateRect(hwnd, nullptr, TRUE); return 0;
    case WM_LBUTTONDOWN: {
        const int mx = GET_X_LPARAM(lp);
        const int my = GET_Y_LPARAM(lp);
        if (add_guide_click(mx, my)) return 0;
        if (toggle_tile_click(mx, my)) return 0;
        return 0;
    }
    case WM_COMMAND: {
        int id = LOWORD(wp);
        int code = HIWORD(wp);
        if (id == IDC_COMBO_SUFFIX && code == CBN_SELCHANGE) {
            sync_suffix_editor_from_combo();
            if (combo_idx(g.combo_channel) == 0) {
                sync_channel_editor_from_combo();
            }
            update_custom_suffix_enabled();
            return 0;
        }
        if (id == IDC_COMBO_CHANNEL && code == CBN_SELCHANGE) {
            sync_channel_editor_from_combo();
            update_custom_channel_enabled();
            return 0;
        }
        if (id == IDC_COMBO_SELECTION_MODE && code == CBN_SELCHANGE) {
            ui_to_state();
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        if (id == IDC_CHECK_SAVE_ALL && code == BN_CLICKED) {
            ui_to_state();
            update_tile_index_enabled();
            return 0;
        }
        if (id == IDC_CHECK_SELECT_TILES && code == BN_CLICKED) {
            ui_to_state();
            update_tile_index_enabled();
            if (g.select_tiles_mode) {
                status(L"Manual tile selection ON. Click tiles on image.");
            } else {
                status(L"Manual tile selection OFF.");
                g.selected_tiles.clear();
            }
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        if (id == IDC_CHECK_SKIP_SOLID && code == BN_CLICKED) {
            ui_to_state();
            return 0;
        }
        if ((id == IDC_EDIT_COLS || id == IDC_EDIT_ROWS || id == IDC_EDIT_EXPAND || id == IDC_EDIT_TILE_INDEX || id == IDC_EDIT_SNAP) &&
            code == EN_CHANGE) {
            ui_to_state();
            update_tile_index_enabled();
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        if (id == IDC_PREV && g.index > 0) { ui_to_state(); --g.index; load_current(); return 0; }
        if (id == IDC_NEXT && g.index + 1 < static_cast<int>(g.files.size())) { ui_to_state(); ++g.index; load_current(); return 0; }
        if (id == IDC_SAVE) { save_current(false); return 0; }
        if (id == IDC_SAVE_NEXT) { save_current(true); return 0; }
        if (id == IDC_OPEN_INPUT) { open_folder(g.input_dir); return 0; }
        if (id == IDC_OPEN_OUTPUT) { ui_to_state(); open_folder(g.output_dir); return 0; }
        if (id == IDC_ADD_VGUIDE) { g.guide_mode = GuideMode::Vertical; status(L"Click image to place vertical guide."); return 0; }
        if (id == IDC_ADD_HGUIDE) { g.guide_mode = GuideMode::Horizontal; status(L"Click image to place horizontal guide."); return 0; }
        if (id == IDC_CLEAR_GUIDES) { g.v_guides.clear(); g.h_guides.clear(); g.guide_mode = GuideMode::None; status(L"Guides cleared."); InvalidateRect(hwnd, nullptr, FALSE); return 0; }
        if (id == IDC_REFRESH) { ui_to_state(); scan_images(); status(L"Input refreshed."); return 0; }
        if (id == IDC_FINISH) {
            ui_to_state();
            fs::path out = g.output_dir.empty() ? fs::path(g.input_dir) / L"manual_corrected" : fs::path(g.output_dir);
            g.output_dir = out.wstring();
            open_folder(g.output_dir);
            PostMessageW(hwnd, WM_CLOSE, 0, 0);
            return 0;
        }
        return 0;
    }
    case WM_PAINT: { PAINTSTRUCT ps{}; auto hdc = BeginPaint(hwnd, &ps); draw(hdc); EndPaint(hwnd, &ps); return 0; }
    case WM_DESTROY: PostQuitMessage(0); return 0;
    default: return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

static bool parse_args(std::wstring& in, std::wstring& out) {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) return false;
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--input" && i + 1 < argc) in = argv[++i];
        else if (a == L"--output" && i + 1 < argc) out = argv[++i];
    }
    LocalFree(argv);
    if (in.empty()) return false;
    if (!fs::exists(fs::path(in)) || !fs::is_directory(fs::path(in))) return false;
    if (out.empty()) out = (fs::path(in) / L"manual_corrected").wstring();
    return true;
}

int WINAPI wWinMain(HINSTANCE hinst, HINSTANCE, LPWSTR, int show) {
    std::wstring in, out;
    if (!parse_args(in, out)) {
        MessageBoxW(nullptr, L"Usage:\nmanual_texture_corrector.exe --input <textures_folder> [--output <folder>]", L"Manual Texture Corrector", MB_OK | MB_ICONINFORMATION);
        return 1;
    }
    g.input_dir = in;
    g.output_dir = out;

    INITCOMMONCONTROLSEX icc{sizeof(INITCOMMONCONTROLSEX), ICC_STANDARD_CLASSES | ICC_UPDOWN_CLASS};
    InitCommonControlsEx(&icc);
    GdiplusStartupInput gsi;
    ULONG_PTR token = 0;
    if (GdiplusStartup(&token, &gsi, nullptr) != Ok) return 1;

    const wchar_t* cls = L"WARNO_MANUAL_TEXTURE_CORRECTOR";
    WNDCLASSW wc{};
    wc.hInstance = hinst;
    wc.lpfnWndProc = wnd_proc;
    wc.lpszClassName = cls;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(0, cls, L"WARNO Manual Texture Corrector", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, 1500, 920, nullptr, nullptr, hinst, nullptr);
    if (!hwnd) { GdiplusShutdown(token); return 1; }
    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    MSG m{};
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    // GDI+ objects must be destroyed before GdiplusShutdown to avoid AV on exit.
    g.bmp.reset();
    GdiplusShutdown(token);
    return static_cast<int>(m.wParam);
}
