import requests
import json
import collections

from bs4 import BeautifulSoup

HEADERS = {
    'Referer': 'http://music.163.com/',
    'Host': 'music.163.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko)'
                  ' Chrome/55.0.2883.95 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}
LYRIC_URL = "http://music.163.com/api/song/lyric?os=pc&id={}&lv=-1&kv=-1&tv=-1"
PLAYLIST_URL = "http://music.163.com/discover/playlist/?order=hot&cat={}&limit=35&offset={}"
PLAYLIST_API = "http://music.163.com/api/playlist/detail?id={}&upd"


def curl(url, headers, dtype='json'):
    s = requests.session()
    soup = BeautifulSoup(s.get(url, headers=headers).content, 'html.parser')
    if dtype == 'json':
        data = json.loads(soup.text)
    elif dtype == 'html':
        data = soup
    else:
        data = soup.text
    return data


class PlaylistInfo(
    collections.namedtuple('PlaylistInfo', (
        'playlist_name', 'playlist_id', 'play_times'
    ))
):
    pass


class MusicInfo(
    collections.namedtuple('MusicInfo', (
        'music_name', 'music_id', 'music_artist'
    ))
):
    pass


def get_playlist(style, offset=0):
    """Get playlists information w.r.t. style.
    Args:
        style: string, music style.
        offset: int, offset of playlists in all playlists,
            a page displays only 35 playlists.
    Return:
        playlist_info: PlaylistInfo."""
    url = PLAYLIST_URL.format(style, offset)
    data = curl(url, headers=HEADERS, dtype='html')
    mf_attr = {'class': 'm-cvrlst f-cb'}
    tag = data.find('ul', mf_attr)
    uu_attr = {'class': 'u-cover u-cover-1'}
    msk_attr = {'class': 'msk'}
    nb_attr = {'class': 'nb'}
    covers = tag.find_all('div', uu_attr)
    playlists = []
    playlist_ids = []
    play_times = []

    for playlist in covers:
        pl_title = playlist.find('a', msk_attr)['title']
        pl_id = playlist.find('a', msk_attr)['href'].replace('/playlist?id=', '')
        pl_cnt = playlist.find('span', nb_attr).text.replace('万', '0000')
        playlists.append(pl_title)
        playlist_ids.append(pl_id)
        play_times.append(pl_cnt)

    return PlaylistInfo(playlist_name=playlists,
                        playlist_id=playlist_ids,
                        play_times=play_times)


def get_music(playlist_id):
    """Get all music in the playlist specified by playlist_id.
    Args:
        playlist_id: int or string.
    Return:
        MusicInfo."""
    url = PLAYLIST_API.format(str(playlist_id))
    data = curl(url, headers=HEADERS, dtype='json')
    tracks = data['result']['tracks']
    music_names = []
    music_ids = []
    music_artists = []
    for track in tracks:
        music_names.append(track['name'])
        music_ids.append(track['id'])
        music_artists.append(track['artists'][0]['name'])

    return MusicInfo(music_name=music_names,
                     music_id=music_ids,
                     music_artist=music_artists)


def get_lyric(song_id):
    """Get lyric of the song specified by song_id.
    Args:
        song_id: int or string.
    Return:
        lyric: string."""
    url = LYRIC_URL.format(str(song_id))
    data = curl(url, HEADERS)
    lyric = data['lrc']['lyric']
    return lyric


if __name__ == '__main__':
    playlist_info = get_playlist('说唱', offset=0)
    idx = playlist_info.playlist_name.index('中国饶舌')
    one_playlist_id = playlist_info.playlist_id[idx]
    music_info = get_music(one_playlist_id)
    for music_id in music_info.music_id:
        lrc = get_lyric(music_id)
        print(lrc)
        break



